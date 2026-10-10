import torch
from torch import nn
from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class DistilBertWithVelocity(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, velocity=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        if velocity is None:
            velocity = torch.zeros((pooled.shape[0], 1), device=pooled.device)
        combined = torch.cat([pooled, velocity.reshape(-1, 1)], dim=1)
        logits = self.classifier(self.dropout(combined))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
