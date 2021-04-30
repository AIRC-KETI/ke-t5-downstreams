
import os

from transformers import T5EncoderModel, T5Config
import torch
import torch.nn.functional as F
from torch import nn

from torchcrf import CRF

class T5EncoderCRF(nn.Module):
    def __init__(self, 
            num_classes, 
            pretrained_path=None,
            cfg_path=None,
            return_dict=True,
            dropout_prob=0.1):
        super(T5EncoderCRF, self).__init__()

        if pretrained_path:
            t5encoder = T5EncoderModel.from_pretrained(pretrained_path, return_dict=return_dict)
        elif cfg_path:
            cfg = T5Config.from_pretrained(cfg_path)
            t5encoder = T5EncoderModel(cfg)
        else:
            raise ValueError('You have to specify either pretrained_path or cfg_path')

        self.t5encoder = t5encoder
        hidden_size = t5encoder.config.d_model
        self.dropout = nn.Dropout(dropout_prob)
        self.position_wise_ff = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, inputs, tags=None):

        outputs = self.t5encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_encoder_layer = outputs.last_hidden_state
        last_encoder_layer = self.dropout(last_encoder_layer)
        emissions = self.position_wise_ff(last_encoder_layer)

        if tags is not None:
            mask = tags["attention_mask"].to(torch.uint8)
            log_likelihood = self.crf(emissions, tags["input_ids"], mask=mask)
            sequence_of_tags = self.crf.decode(emissions, mask)
            return log_likelihood, sequence_of_tags
        else:
            mask = inputs["attention_mask"].to(torch.uint8)
            sequence_of_tags = self.crf.decode(emissions)
            return sequence_of_tags

