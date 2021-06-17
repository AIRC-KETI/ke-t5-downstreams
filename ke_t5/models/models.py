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

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, T5EncoderModel, T5PreTrainedModel, T5ForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from torchcrf import CRF

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
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
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
            if c is None:
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
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
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


@register_model('T5EncoderForTokenClassification')
class T5EncoderForTokenClassification(T5EncoderModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@register_model('T5EncoderForEntityRecognitionWithCRF')
class T5EncoderForEntityRecognitionWithCRF(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(T5EncoderForEntityRecognitionWithCRF, self).__init__(config)

        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.dropout_rate)
        self.position_wise_ff = nn.Linear(config.d_model, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

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

        if attention_mask is None:
            attention_mask = torch.ones(input_ids, device=input_ids.device)

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
        last_hidden_state = self.dropout(last_hidden_state)
        emissions = self.position_wise_ff(last_hidden_state)

        loss = None
        if labels is not None:
            mask = attention_mask.to(torch.uint8)
            loss = self.crf(emissions, labels, mask=mask)
            loss = -1 * loss
            logits = self.crf.decode(emissions, mask)
        else:
            mask = attention_mask.to(torch.uint8)
            logits = self.crf.decode(emissions, mask)

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else logits

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@register_model('T5EncoderForSequenceClassificationRE')
class T5EncoderForSequenceClassificationRE(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(T5EncoderForSequenceClassificationRE, self).__init__(config)
        self.num_labels = config.num_labels        
        self.model_dim = config.d_model
        
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim * 3 ,self.num_labels)
                                       )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_token_idx=None
    ):

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
        pooled_out = torch.mean(last_hidden_state, 1)

        # entity_token_idx = [[sub_start_idx, sub_end_idx],[obj_start_idx, obj_end_idx]]
        # print(entity_token_idx)
        # print(type(entity_token_idx))
        sub_output = []
        obj_output = []
        for b_idx, entity_idx in enumerate(entity_token_idx):
            # print(entity_idx)
            sub_entity_idx, obj_entity_idx = entity_idx
            # print(last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:])
            sub_hidden = last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:]
            sub_hidden_mean = torch.mean(sub_hidden, 0)
            sub_output.append(sub_hidden_mean.unsqueeze(0))
            
            obj_hidden = last_hidden_state[b_idx,obj_entity_idx[0]:obj_entity_idx[1],:]
            obj_hidden_mean = torch.mean(obj_hidden, 0)
            obj_output.append(obj_hidden_mean.unsqueeze(0))

        # for i in range(len(pooled_out)):
        #     sub_hidden = last_hidden_state[i,entities_idx[i,0]:entities_idx[i,1],:]
        #     sub_hidden_mean = torch.mean(sub_hidden, 0)
        #     sub_output.append(sub_hidden_mean.unsqueeze(0))
            
        #     obj_hidden = last_hidden_state[i,entities_idx[i,2]:entities_idx[i,3],:]
        #     obj_hidden_mean = torch.mean(sub_hidden, 0)
        #     obj_output.append(obj_hidden_mean.unsqueeze(0))
            
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))

        entities_concat = torch.cat([pooled_out, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1)
        
        logits = self.classifier(entities_concat)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@register_model('T5EncoderForSequenceClassificationMeanSubmeanObjmean')
class T5EncoderForSequenceClassificationMeanSubmeanObjmean(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(T5EncoderForSequenceClassificationMeanSubmeanObjmean, self).__init__(config)
        self.num_labels = config.num_labels        
        self.model_dim = config.d_model

        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim * 3 ,self.num_labels)
                                       )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_token_idx=None
    ):

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

        sub_output = []
        obj_output = []
        for b_idx, entity_idx in enumerate(entity_token_idx):
            sub_entity_idx, obj_entity_idx = entity_idx
            sub_hidden = last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:]
            sub_hidden_mean = torch.mean(sub_hidden, 0)
            sub_output.append(sub_hidden_mean.unsqueeze(0))
            
            obj_hidden = last_hidden_state[b_idx,obj_entity_idx[0]:obj_entity_idx[1],:]
            obj_hidden_mean = torch.mean(obj_hidden, 0)
            obj_output.append(obj_hidden_mean.unsqueeze(0))
            
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))

        entities_concat = torch.cat([pooled_output, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1)
        
        logits = self.classifier(entities_concat)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 1st_mean + sub_start + obj_start
# 1st_mean + sub_mean + ohb_mean
# 1st_mean + sub_start_end + obj_start_end

# 1st + sub_start + obj_start
# 1st + sub_mean + ohb_mean
# 1st + sub_start_end + obj_start_end

@register_model('T5EncoderForSequenceClassificationREAttention')
class T5EncoderForSequenceClassificationREAttention(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(T5EncoderForSequenceClassificationREAttention, self).__init__(config)
        self.num_labels = config.num_labels        
        self.model_dim = config.d_model
        
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim * 3 ,self.num_labels)
                                       )

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_token_idx=None
    ):

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
        pooled_out = torch.mean(last_hidden_state, 1)



        # last_hidden_state : [b, seq_len, d_model]
        # entity_token_idx : [b, num_entity, idx] = [[sub_start, sub_end], [obj_start, obj_end], ...]
        print('entity_token_idx {}'.format(entity_token_idx))
        # print('entity_token_idx.shape {}'.format(entity_token_idx.shape))
        print('input_ids {}'.format(input_ids))
        print('input_ids.shape {}'.format(input_ids.shape))

        

        # entity_token_idx = [[sub_start_idx, sub_end_idx],[obj_start_idx, obj_end_idx]]
        # print(entity_token_idx)
        # print(type(entity_token_idx))
        sub_output = []
        obj_output = []
        for b_idx, entity_idx in enumerate(entity_token_idx):
            # print(entity_idx)
            sub_entity_idx, obj_entity_idx = entity_idx
            # print(last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:])
            print('sub_entity_idx', sub_entity_idx)
            print('obj_entity_idx',obj_entity_idx)
            sub_hidden = last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:]
            # sub_hidden_mean [d_model] 
            sub_hidden_mean = torch.mean(sub_hidden, 0)

            sub_output.append(sub_hidden_mean.unsqueeze(0))
            
            obj_hidden = last_hidden_state[b_idx,obj_entity_idx[0]:obj_entity_idx[1],:]
            obj_hidden_mean = torch.mean(obj_hidden, 0)
            obj_output.append(obj_hidden_mean.unsqueeze(0))

        # sub_representation : [b, d_model]
        sub_representation = torch.cat((sub_output))
        obj_representation = torch.cat((obj_output))

        # [b, seq_len, d_model] * [b, d_model , 1] = [b, seq_len, 1] -> [b, seq_len]
        sub_attention = torch.bmm(last_hidden_state, sub_representation.unsqueeze(2)).squeeze()
        obj_attention = torch.bmm(last_hidden_state, obj_representation.unsqueeze(2)).squeeze()

        #attention value normalization
        
        print('sub_attention {}'.format(sub_attention))
        print('sub_attention.shape {}'.format(sub_attention.shape))
        
        sub_attention = self.softmax(sub_attention)
        obj_attention = self.softmax(obj_attention)

        # final_atteion : [b, seq_len]
        final_attention = sub_attention * obj_attention
        

        
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))

        entities_concat = torch.cat([pooled_out, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1)
        
        logits = self.classifier(entities_concat)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@register_model('T5EncoderForSequenceClassificationREMeanS2')
class T5EncoderForSequenceClassificationREMeanS2(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(T5EncoderForSequenceClassificationREMeanS2, self).__init__(config)
        self.num_labels = config.num_labels        
        self.model_dim = config.d_model
        
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim * 3 ,self.num_labels)
                                       )

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_token_idx=None
    ):

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
        pooled_out = torch.mean(last_hidden_state, 1)

        # entity_token_idx = [[sub_start_idx, sub_end_idx],[obj_start_idx, obj_end_idx]]
        # print(entity_token_idx)
        # print(type(entity_token_idx))
        sub_output = []
        obj_output = []
        for b_idx, entity_idx in enumerate(entity_token_idx):
            # print(entity_idx)
            sub_entity_idx, obj_entity_idx = entity_idx
            # print(last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:])
            sub_hidden = last_hidden_state[b_idx,sub_entity_idx[0]:sub_entity_idx[1],:]
            sub_hidden_mean = torch.mean(sub_hidden, 0)
            sub_output.append(sub_hidden_mean.unsqueeze(0))
            
            obj_hidden = last_hidden_state[b_idx,obj_entity_idx[0]:obj_entity_idx[1],:]
            obj_hidden_mean = torch.mean(sub_hidden, 0)
            obj_output.append(obj_hidden_mean.unsqueeze(0))
            
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))

        entities_concat = torch.cat([pooled_out, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1)
        
        logits = self.classifier(entities_concat)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )