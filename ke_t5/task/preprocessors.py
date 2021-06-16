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

import re
import copy


import numpy as np

from ke_t5 import pipe as seq_pipe


_DEFAULT_SPAN_TAGS = ['O', 'B', 'I']


def _collapse_consecutive_spaces(text):
    return re.sub(r'\s+', ' ', text)


def _string_join(lst, sep=' '):
    # Join on space, but collapse consecutive spaces.
    out = sep.join(lst)
    return _collapse_consecutive_spaces(out)


def _pad_punctuation_general(text):
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r'([^_\s\w])', r' \1 ', text)
    # Collapse consecutive whitespace into one space.
    text = _collapse_consecutive_spaces(text)
    return text


@seq_pipe.map_over_dataset
def base_preproc_for_classification(
        x,
        benchmark_name,
        input_keys,
        label_names=None,
        no_label_idx=0,
        with_feature_key=True,
        sep=' '):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = label_names[x['label']] if x['label'] >= 0 else '<unk>'
    else:
        ex['targets'] = x['label'] if x['label'] >= 0 else no_label_idx

    joined = sep.join(strs_to_join)
    ex['inputs'] = joined

    return ex


@seq_pipe.map_over_dataset
def base_preproc_for_regression(
        x,
        benchmark_name,
        input_keys,
        is_string_tgt=True,
        with_feature_key=True,
        sep=' '):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if is_string_tgt:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = "{:.1f}".format(np.round_(x['label'], 1))
    else:
        ex['targets'] = np.round_(x['label'], 1)

    joined = sep.join(strs_to_join)
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seq_pipe.map_over_dataset
def re_preproc_for_classification(
        x,
        benchmark_name,
        label_names=None,
        no_label_idx=0,
        with_feature_key=True,
        sep=' '):
    # mark span using start index of the entity
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:[\S\s]){N})(W)'
        pattern_tmpl = pattern_tmpl.replace('N', str(span_idx))
        pattern = pattern_tmpl.replace('W', span_str)
        return re.sub(pattern, r'\1{0}\2{0}'.format(mark), text)

    # '*' for subejct entity '#' for object entity.

    text = x["sentence"]
    text = _mark_span(text, x['subject_entity']['word'],
                      x['subject_entity']['start_idx'], '*')
    # Compensate for 2 added "words" added in previous step.
    span2_index = x['object_entity']['start_idx'] + 2 * \
        (1 if x['subject_entity']['start_idx'] <
         x['object_entity']['start_idx'] else 0)
    text = _mark_span(text, x['object_entity']['word'], span2_index, '#')

    strs_to_join = []
    if with_feature_key:
        strs_to_join.append('{}:'.format('text'))
    strs_to_join.append(text)

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = label_names[x['label']] if x['label'] >= 0 else '<unk>'
    else:
        ex['targets'] = x['label'] if x['label'] >= 0 else no_label_idx

    joined = sep.join(strs_to_join)
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seq_pipe.map_over_dataset
def base_preproc_for_conditional_generation(
        x,
        prefix,
        input_keys,
        with_feature_key=True,
        sep=' '):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    strs_to_join.insert(0, prefix)

    joined = sep.join(strs_to_join)
    ex['inputs'] = joined

    return ex


@seq_pipe.map_over_dataset
def preprocess_quad(x, benchmark_name, include_context=True, impossible_answer_text='impossible', pad_punct=True):

    a = x['answers']['text']
    q = x['question']
    c = x['context']

    if pad_punct:
        a = [_pad_punctuation_general(txt) for txt in a]
        q = _pad_punctuation_general(q)
        c = _pad_punctuation_general(c)

    strs_to_join = []
    if include_context:
        strs_to_join.extend(['question:', q, 'context:', c])
    else:
        strs_to_join.extend(['trivia question:', q])

    strs_to_join.insert(0, benchmark_name)
    inputs = _string_join(strs_to_join)

    if 'is_impossible' in x:
        if x['is_impossible']:
            label = impossible_answer_text
        else:
            label = a[0]
    else:
        label = a[0]

    return {
        'inputs': inputs,
        'targets': label,
        'id': x['id'],
        'context': c,
        'question': q,
        'answers': a
    }


@seq_pipe.map_over_dataset
def tokenize_and_preproc_iob2(x, output_features, tags=None, iob2_tags=None, tag_label='NE', input_key='inputs', info4klue=True):

    ret = {}

    inputs = x[input_key]
    tokenizer = output_features[input_key].tokenizer
    ret[f'{input_key}_pretokenized'] = inputs
    input_hf = tokenizer(inputs)
    input_ids = input_hf.input_ids

    if info4klue:
        ret['klue_metric'] = {}
        ret['klue_metric']['char_to_token'] = [
            input_hf.char_to_token(pos) for pos in range(len(inputs))]

    ret[f'{input_key}'] = input_ids
    #ret['char_to_token'] = {k:v for k, v in enumerate(char_to_token)}
    #ret[f'{input_key}_tokens'] = [tokenizer._convert_id_to_token(x) for x in input_ids]

    if tags and iob2_tags:
        outside_label = iob2_tags.index('O')
        tag_labels = x[tag_label]
        labels = np.ones_like(input_ids, dtype=np.int32) * outside_label

        if info4klue:
            ret['klue_metric']['char_tag'] = np.ones_like(
                ret['klue_metric']['char_to_token'], dtype=np.int32) * outside_label

        for begin, end, label in zip(tag_labels['begin'], tag_labels['end'], tag_labels['label']):

            label_txt = tags[label]
            if label_txt != 'O':

                if info4klue:
                    for idx, pos_idx in enumerate(list(range(begin, end))):
                        if idx == 0:
                            ret['klue_metric']['char_tag'][pos_idx] = iob2_tags.index(
                                'B-'+label_txt)
                        else:
                            ret['klue_metric']['char_tag'][pos_idx] = iob2_tags.index(
                                'I-'+label_txt)

                pos_list = [input_hf.char_to_token(
                    pos) for pos in range(begin, end)]
                #pos_list = copy.deepcopy(char_to_token[begin:end])

                # there is  None position in the case consecutive white spaces.
                pos_list = [x for x in pos_list if x is not None]
                token_set = set(pos_list)
                token_set_order = sorted(list(token_set))
                for iter_idx, tk_idx in enumerate(token_set_order):
                    if iter_idx == 0:
                        labels[tk_idx] = iob2_tags.index('B-'+label_txt)
                    else:
                        labels[tk_idx] = iob2_tags.index('I-'+label_txt)
        ret['targets'] = labels
    return ret


@seq_pipe.map_over_dataset_batched_format
def create_doc_span(x, 
                    output_features, 
                    span_length=448, 
                    doc_stride=128, 
                    input_key='inputs'):

    ret = {
        'id': [],
        'span_ch_start': [],
        'span_ch_end': [],
    }

    if input_key not in ret:
        ret[input_key] = []
    
    keep_features = [k for k in x.keys() if k not in ret]

    for keep_f in keep_features:
        ret[keep_f] = []

    batch_len = len(x[input_key])

    tokenizer = output_features[input_key].tokenizer

    for ex_idx in range(batch_len):
        inputs = x[input_key][ex_idx]
        input_hf = tokenizer(inputs, add_special_tokens=False)
        input_ids = input_hf.input_ids

        doc_spans = []
        start_offset = 0
        while start_offset < len(input_ids):
            length = len(input_ids) - start_offset
            if length > span_length:
                length = span_length
            doc_spans.append((start_offset, length))
            if start_offset + length == len(input_ids):
                break
            start_offset += min(length, doc_stride)

        for doc_idx, doc_span in enumerate(doc_spans):
            doc_tk_st, doc_tk_ed = doc_span[0], doc_span[0] + doc_span[1] - 1

            doc_ch_st = input_hf.token_to_chars(doc_tk_st).start
            doc_ch_ed = input_hf.token_to_chars(doc_tk_ed).end

            ret['id'].append('{}:{}'.format(x['id'][ex_idx], doc_idx))
            ret['span_ch_start'].append(doc_ch_st)
            ret['span_ch_end'].append(doc_ch_ed)
            ret[input_key].append(inputs[doc_ch_st:doc_ch_ed])
            
            for keep_f in keep_features:
                if keep_f in x:
                    ret[keep_f].append(copy.deepcopy(x[keep_f][ex_idx]))

    return ret


@seq_pipe.map_over_dataset
def tokenize_and_preproc_cr_span_extraction(x, output_features, iob2_tags=_DEFAULT_SPAN_TAGS, input_key='inputs'):
    ret = {}

    inputs = x[input_key]
    tokenizer = output_features[input_key].tokenizer
    ret[f'{input_key}_pretokenized'] = inputs
    input_hf = tokenizer(inputs, add_special_tokens=False)
    input_ids = input_hf.input_ids
    ret[input_key] = input_ids

    outside_label = iob2_tags.index('O')
    labels = np.ones_like(input_ids, dtype=np.int32) * outside_label

    offset = 0
    if 'span_ch_start' in x:
        offset = x['span_ch_start']
    end_offset = len(inputs)
    if 'span_ch_end' in x:
        end_offset = x['span_ch_end']
    
    for mention in x['CR']['mention']:
        for begin, end, form in zip(mention['begin'], mention['end'], mention['form']):
            if offset < begin and end < end_offset:
                begin -= offset
                end -= offset

                pos_list = [input_hf.char_to_token(
                    pos) for pos in range(begin, end)]
                # there is  None position in the case consecutive white spaces.
                pos_list = [x for x in pos_list if x is not None]
                token_set_order = sorted(list(set(pos_list)))

                for iter_idx, tk_idx in enumerate(token_set_order):
                    if iter_idx == 0:
                        labels[tk_idx] = iob2_tags.index('B')
                    else:
                        labels[tk_idx] = iob2_tags.index('I')
            
    ret['targets'] = labels
                
    return ret


def mention_marking(text, mention, mention_cands, mention_marker='*'):
    def _mark_span(text, span_st, span_ed, mark):
        return text[:span_st] + mark + text[span_st:span_ed] + mark + text[span_ed:]
    offset_st = mention[0]
    offset_ed = mention[1]
    text = _mark_span(text, offset_st, offset_ed, mention_marker)
    mention_cands_new = []
    for x, y, z in mention_cands:
        if x > offset_ed:
            mention_cands_new.append((x+2, y+2, z))
        elif y == offset_ed:
            # char span (include marker)
            mention_cands_new.append((x, y+2, z))
        else:
            mention_cands_new.append((x, y, z))

    return text, mention_cands_new


def create_labels_for_mention(text, mention_cands, tokenizer, iob2_tags=_DEFAULT_SPAN_TAGS):
    input_hf = tokenizer(text, add_special_tokens=False)
    input_ids = input_hf.input_ids

    outside_label = iob2_tags.index('O')
    labels = np.ones_like(input_ids, dtype=np.int32) * outside_label

    label_txt = []

    for begin, end, form in mention_cands:
        pos_list = [input_hf.char_to_token(
            pos) for pos in range(begin, end)]
        # there is  None position in the case consecutive white spaces.
        pos_list = [x for x in pos_list if x is not None]
        token_set_order = sorted(list(set(pos_list)))

        label_txt.append(tokenizer.decode([input_ids[x] for x in token_set_order]))

        for iter_idx, tk_idx in enumerate(token_set_order):
            if iter_idx == 0:
                labels[tk_idx] = iob2_tags.index('B')
            else:
                labels[tk_idx] = iob2_tags.index('I')
    
    return input_ids, labels, label_txt

@seq_pipe.map_over_dataset_batched_format
def create_cr_example(x, 
                    output_features,
                    exclude_src_span=False,
                    iob2_tags=_DEFAULT_SPAN_TAGS,
                    input_key='inputs'):
    ret = {
        'id': [],
        'targets': [],
        f'{input_key}_pretokenized': [],
    }

    if input_key not in ret:
        ret[input_key] = []
    
    keep_features = [k for k in x.keys() if k not in ret]

    for keep_f in keep_features:
        ret[keep_f] = []

    batch_len = len(x[input_key])

    tokenizer = output_features[input_key].tokenizer

    for ex_idx in range(batch_len):
        inputs = x[input_key][ex_idx]

        offset = 0
        if 'span_ch_start' in x:
            offset = x['span_ch_start'][ex_idx]
        end_offset = len(inputs)
        if 'span_ch_end' in x:
            end_offset = x['span_ch_end'][ex_idx]
        
        for mention in x['CR'][ex_idx]['mention']:
            valid_mensions = [(begin-offset, end-offset, form) 
                for begin, end, form 
                in zip(mention['begin'], mention['end'], mention['form'])
                if offset < begin and end < end_offset]
            
            for mention_idx in range(len(valid_mensions)):
                mention_tgt = valid_mensions[mention_idx]

                if exclude_src_span:
                    mention_cands = [valid_mensions[i] for i in range(len(valid_mensions)) if i != mention_idx]
                else:
                    mention_cands = valid_mensions

                inputs_aug, mention_cand_aug = mention_marking(inputs, mention_tgt, mention_cands)
                input_ids_aug, labels_aug, label_txt = create_labels_for_mention(inputs_aug, mention_cand_aug, tokenizer)

                #print(inputs_aug, input_ids_aug, labels_aug, mention_cand_aug, label_txt)

                ret[f'{input_key}_pretokenized'].append(inputs_aug)
                ret[input_key].append(input_ids_aug)
                ret['targets'].append(labels_aug)
                ret['id'].append('{}:{}'.format(x['id'][ex_idx], mention_idx))

                for keep_f in keep_features:
                    if keep_f in x:
                        ret[keep_f].append(copy.deepcopy(x[keep_f][ex_idx]))

    return ret


