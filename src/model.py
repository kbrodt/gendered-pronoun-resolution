import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class PairScore(nn.Module):
    def __init__(self, input_dim, hidden_dim, p):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, g1, g2):
        return self.score(torch.cat([g1, g2, g1*g2], dim=-1))


class GAP(nn.Module):
    def __init__(self, bert_model, input_dim, hidden_dim, p=0.8, last_layers=4, fine_tune=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        for param in self.bert.parameters():
            param.requires_grad = fine_tune
        self.last_layers = last_layers
        self.pair_score = PairScore(3*input_dim*self.last_layers, hidden_dim, p=p)
    
    def forward(self, token_tensor, offsets):
        bert_outputs, _ = self.bert(
            token_tensor,
            attention_mask=(token_tensor > 0).long(), 
            token_type_ids=None,
            output_all_encoded_layers=True)
        
        bert_outputs = torch.cat(bert_outputs[-self.last_layers:], dim=-1)
        
        bert_output = bert_outputs.gather(1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2)))
        a, b, p = torch.unbind(bert_output, dim=1)
        
        ap_score = self.pair_score(p, a)
        bp_score = self.pair_score(p, b)
        nan_score = torch.zeros_like(ap_score)
        logits = torch.cat([ap_score, bp_score, nan_score], dim=1)
        
        return logits