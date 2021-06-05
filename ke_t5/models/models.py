# Copyright 2021 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5EncoderModel, T5PreTrainedModel

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput

from .loader import register_model


class SimplePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense1(first_token_tensor)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)
        
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
        divisor = mask.sum(dim=1).view(-1, 1).float()
        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = self.dense1(sentence_sums)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)
        
        return pooled_output


@register_model('T5EncoderForSequenceClassificationSimple')
class T5EncoderForSequenceClassificationSimple(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationSimple, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model('T5EncoderForSequenceClassificationMean')
class T5EncoderForSequenceClassificationMean(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderForSequenceClassificationMean, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




