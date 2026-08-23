import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, prompts, device):
        """
        prompts: list of strings
        Returns: token_embeddings [B, L, D], attention_mask [B, L]
        """
        tokenized = self.tokenizer(prompts, padding=True, truncation=True,
                                     return_tensors="pt", max_length=16).to(device)
        if self.freeze:
            with torch.no_grad():
                output = self.bert(**tokenized)
        else:
            output = self.bert(**tokenized)
        return output.last_hidden_state, tokenized["attention_mask"]
