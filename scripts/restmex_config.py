params = {
    "experiment_name": "MTL_RESTMEX",
    "dataset_version": "",
    "seed": 43,
    "model": {
        "name": "pysentimiento/robertuito-base-cased",
        "n_labels_type": 3,
        "n_labels_town": 40,
        "n_labels_polarity": 5,
        "dropout": 0.01
    },
    "train": {
        "epochs": 5,
        "batch_size": 5,
        "gradient_accumulator_size": 2,
        "optimizer": {
            "lr": 1e-5,
            "eps": 1e-8,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "amsgrad": False
        },
        "criterion": {
            "type":{
                "reduction": "mean", #sum
                "label_smoothing": 0    
            },
            "town":{
                "reduction": "mean", #sum
                "label_smoothing": 0    
            },
            "polarity":{
                "reduction": "mean", #sum  
            },
        },    
        "scheduler":{
            "num_cycles": 1.5
        }
    },
    "tokenizer":{
        "max_length": 128,
        "padding": 'max_length',
        "truncation": True
    }
}

def flatten_dict(d=params, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items