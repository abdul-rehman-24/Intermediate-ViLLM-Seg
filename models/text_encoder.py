import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 freeze=True, unfreeze_last_n_layers=0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.freeze = freeze and unfreeze_last_n_layers == 0

        if freeze and unfreeze_last_n_layers > 0:
            for param in self.bert.parameters():
                param.requires_grad = False
            total_layers = len(self.bert.encoder.layer)
            for i in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            print(f"Unfroze last {unfreeze_last_n_layers} of {total_layers} BERT layers")
        elif freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, prompts, device):
        tokenized = self.tokenizer(prompts, padding=True, truncation=True,
                                     return_tensors="pt", max_length=16).to(device)
        if self.freeze:
            with torch.no_grad():
                output = self.bert(**tokenized)
        else:
            output = self.bert(**tokenized)
        return output.last_hidden_state, tokenized["attention_mask"]
