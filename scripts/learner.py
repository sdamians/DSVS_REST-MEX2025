import time
import torch as t
import torch.nn as nn
import numpy as np
import gc
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from utils import format_time
from utils import get_metrics

class Learner:
  def __init__(self, classifier, 
               optimizer_params: dict, 
               scheduler_params: dict, 
               class_weights_type: Tensor,
               class_weights_polarity: Tensor|None,
               class_weights_town: Tensor,
               criterion_params_type: dict,
               criterion_params_polarity: dict|None,
               criterion_params_town: dict,
               device="cpu"):
    
    self.model = classifier
    self.optimizer = t.optim.Adam(self.model.parameters(), **optimizer_params)
    self.scheduler_params = scheduler_params
    self.device = device

    self.criterion_type = nn.CrossEntropyLoss(weight=class_weights_type, **criterion_params_type)
    self.criterion_polarity = OrdinalLoss(class_weights_polarity, criterion_params_polarity) 
    self.criterion_town = nn.CrossEntropyLoss(weight=class_weights_town, **criterion_params_town)    

  def train(self, trainset: DataLoader, valset: DataLoader, n_epochs: int, gradient_accumulator_size: int=2):
    t_gral = time.time()

    max_step_t = len(trainset)
    total_training_steps = (len(trainset.dataset) // (trainset.batch_size * gradient_accumulator_size)) * n_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),  # Warmup del 10%
        num_training_steps=(total_training_steps),
        **self.scheduler_params  # Opcional: media onda de coseno (default)
    )

    scaler = t.amp.GradScaler(device=self.device)

    # Training mode.
    self.model.train()
    self.model.zero_grad()
  
    for epoch in range(n_epochs):
      t0 = time.time() # We save the start time to see how long it takes.
      epoch_loss = []  # We reset the loss value for each epoch.
      
      with tqdm(total=len(trainset), desc=f'Epoch {epoch + 1}/{n_epochs}', dynamic_ncols=True) as pbar:
        for step, batch in enumerate(trainset):
          batch_loss = 0

          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          labels_town = batch["label_town"].to(self.device)
          labels_polarity = batch["label_polarity"].to(self.device)
          labels_type = batch["label_type"].to(self.device)

          with t.amp.autocast(device_type=self.device):
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # We calculate the loss of the present minibatch
            loss_town = self.criterion_town(outputs["logits_town"], labels_town) 
            loss_type = self.criterion_type(outputs["logits_type"], labels_type)
            loss_polarity = self.criterion_polarity(outputs["logits_polarity"], labels_polarity) 
            loss = loss_type + loss_town + loss_polarity

            batch_loss += loss.item()
            pbar.set_postfix({ "town": loss_town.item(), "type": loss_type.item(), "polarity": loss_polarity.item() })
            pbar.update(1)

          # Backpropagation
          scaler.scale(loss).backward()  # loss.backward()
          
          # So we can implement gradient accumulator technique
          if (step > 0 and step % gradient_accumulator_size == 0) or (step == max_step_t - 1):

            #(this prevents the gradient from becoming explosive)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update learning rate each end of epoch
            # We update the weights and bias according to the optimizer
            scaler.step(self.optimizer)    # self.optimizer.step()
            scaler.update()
            scheduler.step()
            
            # We clean the gradients for the accumulator batch
            self.model.zero_grad()

          input_ids.to("cpu")
          attention_mask.to("cpu")
          labels_town.to("cpu")
          labels_type.to("cpu")
          labels_polarity.to("cpu")

          del input_ids
          del attention_mask
          del labels_town
          del labels_type
          del labels_polarity

          t.cuda.empty_cache()
          gc.collect()

          # if (step % ( max_step_t // 5 ) == 0) or (step == max_step_t - 1):
          #   print(f"Batch {step}/{max_step_t} avg loss: {np.sum(epoch_loss) / (step+1):.5f}")
        
          epoch_loss.append(batch_loss)

      # We calculate the average loss in the current epoch of the training set
      print(f"\n\tAverage training loss: {np.sum(epoch_loss)/max_step_t:.4f}")
      print(f"\tTraining epoch {epoch + 1} took: {format_time(time.time() - t0)}")

      if valset is not None:
        print("\n\tValidation step:")
        self.test(trainset, "trainset metrics:")
        self.test(valset, "valset metrics:")

    print(f"\nTraining complete. It took: {format_time(time.time() - t_gral)}")

  def test(self, testset: DataLoader, msg: str):
    t0 = time.time()

    # We put the model in validation mode
    self.model.eval()

    # We declare variables
    val_town = { "probs":[], "preds":[], "labels":[] }
    val_type = { "probs":[], "preds":[], "labels":[] }
    val_polarity = { "probs":[], "preds":[], "labels":[] }

    with t.no_grad():
      for batch in testset:

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels_town = batch["label_town"].to(self.device) if "label_town" in batch else None
        labels_type = batch["label_type"].to(self.device) if "label_type" in batch else None
        labels_polarity = batch["label_polarity"].to(self.device) if "label_polarity" in batch else None

        outputs = self.model(input_ids, attention_mask=attention_mask)

        probs_town = t.softmax(outputs["logits_town"], dim=-1) # [batch n_classes]
        probs_type = t.softmax(outputs["logits_type"], dim=-1) # [batch n_classes]
        probs_polarity = t.softmax(outputs["logits_polarity"], dim=-1) # [batch n_classes]
        
        preds_town = t.argmax(probs_town, dim=1).unsqueeze(-1) # [batch 1]
        preds_type = t.argmax(probs_type, dim=1).unsqueeze(-1) # [batch 1]
        preds_polarity = t.argmax(probs_polarity, dim=1).unsqueeze(-1) # [batch 1]

        val_town["probs"].extend(probs_town.cpu().numpy())
        val_type["probs"].extend(probs_type.cpu().numpy())
        val_polarity["probs"].extend(probs_polarity.cpu().numpy())

        val_town["preds"].extend(preds_town.cpu().numpy())
        val_type["preds"].extend(preds_type.cpu().numpy())
        val_polarity["preds"].extend(preds_polarity.cpu().numpy())

        if labels_town is not None:
          val_town["labels"].extend(labels_town.cpu().numpy())
          val_type["labels"].extend(labels_type.cpu().numpy())
          val_polarity["labels"].extend(labels_polarity.cpu().numpy())
          

    # We show the final metric scores
    val_town["probs"] = np.array(val_town["probs"])
    val_type["probs"] = np.array(val_type["probs"])
    val_polarity["probs"] = np.array(val_polarity["probs"])

    val_town["preds"] = np.array(val_town["preds"]).flatten()
    val_type["preds"] = np.array(val_type["preds"]).flatten()
    val_polarity["preds"] = np.array(val_polarity["preds"]).flatten()
    
    if labels_town is not None:
      val_town["labels"] = np.array(val_town["labels"]).flatten()
      val_type["labels"] = np.array(val_type["labels"]).flatten()
      val_polarity["labels"] = np.array(val_polarity["labels"]).flatten()

      metrics_town = get_metrics(val_town["labels"], val_town["preds"], None, promedio='macro')
      metrics_type = get_metrics(val_type["labels"], val_type["preds"], None, promedio='macro')
      metrics_polarity = get_metrics(val_polarity["labels"], val_polarity["preds"], None, promedio='macro')

      print(msg)
      print(f"\ttown f1-score: {metrics_town['f1_score']}")
      print(f"\ttype f1-score: {metrics_type['f1_score']}")
      print(f"\tpolarity f1-score: {metrics_polarity['f1_score']}")
      
      print(f"\tValidation took: {format_time(time.time() - t0)}")

    return val_town, val_type, val_polarity
  

class OrdinalLoss(nn.Module):
  def __init__(self, weights, criterion_params):
    super().__init__()
    self.criterion_params = criterion_params
    self.weights = weights

  def forward(self, logits, labels):
    """
    logits: [ batch_size, num_classes - 1 ]
    labels: [ batch_size ] - con valores 0 - (num_classes-1)
    """
    batch_size, num_outputs = logits.shape
    cumulative_labels = t.zeros((batch_size, num_outputs), dtype=t.float32).to(logits.device)

    for k in range(num_outputs):
        cumulative_labels[:, k] = (labels > k).float()

    loss = F.binary_cross_entropy_with_logits(logits, cumulative_labels, **self.criterion_params)
    return loss