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
import numpy as np

from ke_t5 import pipe as seq_pipe


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
    ex['id'] = x['id']

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
    span2_index = x['object_entity']['start_idx'] + 2 * (1 if x['subject_entity']['start_idx'] < x['object_entity']['start_idx'] else 0)
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
def tokenize_and_preproc_iob2(x, output_features, tags=None, iob2_tags=None, tag_label='NE', input_key='inputs'):

    ret = {}

    inputs = x[input_key]
    tokenizer = output_features[input_key].tokenizer
    ret[f'{input_key}_pretokenized'] = inputs
    input_hf = tokenizer(inputs)
    input_ids = input_hf.input_ids
    ret[f'{input_key}'] = input_ids
    ret[f'{input_key}_tokens'] = [tokenizer._convert_id_to_token(x) for x in input_ids]

    if tags and iob2_tags:
        outside_label = iob2_tags.index('O')
        tag_labels = x[tag_label]
        labels = np.ones_like(input_ids, dtype=np.int32) * outside_label

        for begin, end, label in zip(tag_labels['begin'], tag_labels['end'], tag_labels['label']):
            label_txt = tags[label]
            if label_txt != 'O':
                pos_list = [input_hf.char_to_token(pos) for pos in range(begin, end)]

                is_none = False
                for index in pos_list:
                    if index is None:
                        is_none = True
                if is_none:
                    print(x)

                # there is  None position in the case consecutive white spaces.
                pos_list = [x for x in pos_list if x is not None]
                token_set = set(pos_list)
                token_set_order = sorted(list(token_set))
                for iter_idx, tk_idx in enumerate(token_set_order):
                    if iter_idx == 0:
                        labels[tk_idx] = iob2_tags.index('B-'+label_txt)
                    else:
                        labels[tk_idx] = iob2_tags.index('I-'+label_txt)
        ret['labels'] = labels
    return ret
