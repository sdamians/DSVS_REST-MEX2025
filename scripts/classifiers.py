import torch as t
import torch.nn as nn
from transformers import AutoModel

#Multitask learning
class MTLClassifier(nn.Module):
    def __init__(self, modelpath, n_labels_type=2, n_labels_polarity=2, n_labels_town=2, dropout=0.05, device='cpu'):
        super().__init__()

        self.llm = AutoModel.from_pretrained(modelpath
                                            ,output_attentions=False
                                            ,output_hidden_states=False).to(device)

        self.dropout = nn.Dropout(dropout)
        
        self.fc_type = nn.Linear(self.llm.config.hidden_size, n_labels_type)
        self.fc_polarity = nn.Linear(self.llm.config.hidden_size, n_labels_polarity)
        self.fc_town = nn.Linear(self.llm.config.hidden_size, n_labels_town)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids
                            ,token_type_ids=None
                            ,attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        
        logits1 = self.fc_type(pooled_output)
        logits2 = self.fc_polarity(pooled_output)
        logits3 = self.fc_town(pooled_output)

        return { "logits_type": logits1, "logits_polarity": logits2, "logits_town": logits3 }