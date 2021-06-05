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

import seqio
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import batch_matrix_band_part


def _pad_punctuation_kor(text):
    # Add space around punctuation. Hangul Syllable(\uAC00-\uD7AF)
    text = tf.strings.regex_replace(
        text, '([^A-Za-z0-9_\uAC00-\uD7AF])', r' \1 ')
    # Collapse consecutive whitespace into one space.
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text

# imported from multilingual t5 github


def _pad_punctuation_general(text):
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = tf.strings.regex_replace(text, r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ')
    # Collapse consecutive whitespace into one space.
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    return text

# imported from multilingual t5 github


def _string_join(lst):
    # Join on space, but collapse consecutive spaces.
    out = tf.strings.join(lst, separator=' ')
    return tf.strings.regex_replace(out, r'\s+', ' ')


@seqio.map_over_dataset
def base_preproc_for_classification(x, benchmark_name, input_keys, label_names=None, no_label_idx=0, with_feature_key=True):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use "<unk>"
            tf.equal(x['label'], -1),
            lambda: tf.constant('<unk>'),
            # Otherwise grab the label text from label_names
            lambda: tf.gather(label_names, x['label']),
        )
    else:
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use no_label_idx
            tf.equal(x['label'], -1),
            lambda: tf.constant(no_label_idx, tf.int64),
            # Otherwise set the label
            lambda: x['label'],
        )

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def base_preproc_for_regression(x, benchmark_name, input_keys, is_string_tgt=True, with_feature_key=True):
    strs_to_join = []
    for key in input_keys:
        if with_feature_key:
            strs_to_join.append('{}:'.format(key))
        strs_to_join.append(x[key])

    ex = {}

    if is_string_tgt:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
    else:
        ex['targets'] = x['label']

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def re_preproc_for_classification(x, benchmark_name, label_names=None, no_label_idx=0, with_feature_key=True):
    # mark span using start index of the entity
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:[\S\s]){N})(W)'
        pattern = tf.strings.regex_replace(pattern_tmpl, 'N',
                                           tf.as_string(span_idx))
        pattern = tf.strings.regex_replace(pattern, 'W', span_str)
        return tf.strings.regex_replace(text, pattern, r'\1{0}\2{0}'.format(mark))

    # '*' for subejct entity '#' for object entity.

    text = x["sentence"]
    text = _mark_span(text, x['subject_entity']['word'],
                      x['subject_entity']['start_idx'], '*')
    # Compensate for 2 added "words" added in previous step.
    span2_index = x['object_entity']['start_idx'] + 2 * tf.cast(
        x['subject_entity']['start_idx'] < x['object_entity']['start_idx'], tf.int32)
    text = _mark_span(text, x['object_entity']['word'], span2_index, '#')

    strs_to_join = []
    if with_feature_key:
        strs_to_join.append('{}:'.format('text'))
    strs_to_join.append(text)

    ex = {}

    if label_names is not None:
        # put the name of benchmark if the model is generative
        strs_to_join.insert(0, benchmark_name)
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use "<unk>"
            tf.equal(x['label'], -1),
            lambda: tf.constant('<unk>'),
            # Otherwise grab the label text from label_names
            lambda: tf.gather(label_names, x['label']),
        )
    else:
        ex['targets'] = tf.cond(
            # When no label is provided (label == -1), use no_label_idx
            tf.equal(x['label'], -1),
            lambda: tf.constant(no_label_idx, tf.int64),
            # Otherwise set the label
            lambda: x['label'],
        )

    joined = tf.strings.join(strs_to_join, separator=' ')
    ex['inputs'] = joined
    ex['id'] = x['id']

    return ex


@seqio.map_over_dataset
def preprocess_quad(x, benchmark_name, include_context=True, impossible_answer_text='impossible'):
    a = _pad_punctuation_general(x['answers']['text'])
    q = _pad_punctuation_general(x['question'])
    c = _pad_punctuation_general(x['context'])

    strs_to_join = []
    if include_context:
        strs_to_join.extend(['question:', q, 'context:', c])
    else:
        strs_to_join.extend(['trivia question:', q])

    strs_to_join.insert(0, benchmark_name)
    inputs = _string_join(strs_to_join)

    if 'is_impossible' in x:
        if x['is_impossible']:
            label = tf.constant(impossible_answer_text)
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

def tokenize_with_offsets(
        dataset: tf.data.Dataset,
        output_features: seqio.preprocessors.OutputFeaturesType,
        copy_pretokenized: bool = True,
    ) -> tf.data.Dataset:

    def _tokenize(features):
        ret = {}
        for k, v in features.items():
            if k in output_features:
                if copy_pretokenized:
                    ret[f'{k}_pretokenized'] = v
                vocab = output_features[k].vocabulary
                (tokens, start_offsets, end_offsets) = vocab.tf_tokenizer.tokenize_with_offsets(v)
                v = tokens
                ret[f'{k}_start_offsets'] = start_offsets
                ret[f'{k}_end_offsets'] = end_offsets
            ret[k] = v
        return ret

    return dataset.map(
        _tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# @seqio.map_over_dataset
# def preprocess_iob2(x, tag_key='inputs', tags=None, iob2tags=None):
#     tags_to_ib2tags = []
#     for tag in tags:
#         btag = 'B-{}'.format(tag)
#         itag = 'I-{}'.format(tag)
#         tags_to_ib2tags.append([iob2tags.index(btag), iob2tags.index(itag)])
    
#     inputs = x[tag_key]
#     inputs_start_offsets = x[f'{tag_key}_start_offsets']
#     inputs_end_offsets = x[f'{tag_key}_end_offsets']
#     inputs_offsets = tf.concat(
#         [
#             tf.cast(tf.expand_dims(inputs_start_offsets, -1), dtype=tf.int32),
#             tf.cast(tf.expand_dims(inputs_end_offsets, -1), dtype=tf.int32)
#         ],
#         -1
#     )

#     ne_tags = tf.concat(
#         [
#             tf.cast(tf.expand_dims(x['NE']['begin'], -1), dtype=tf.int32),
#             tf.cast(tf.expand_dims(x['NE']['end'], -1), dtype=tf.int32),
#             tf.cast(tf.expand_dims(x['NE']['label'], -1), dtype=tf.int32)
#         ],
#         -1
#     )

#     ne_length = tf.shape(ne_tags)[0] # number of ne tags
#     ne_loop_cond = lambda i, result, offsets, tags, out_idx: tf.less(i, ne_length) # until number of ne tags
#     def ne_loop(ne_idx, result, offsets, tags, out_idx):

#         def offset_loop(offset_idx, result2, offsets2, tags2, is_first):
#             st_offset = offsets2[offset_idx][0]
#             ed_offset = offsets2[offset_idx][1]
#             st_tags = tags2[0]
#             ed_tags = tags2[1]
#             lbl_tags = tags2[2]

#             def check_first(lbl_tags, is_first):
#                 lbl = tf.cond(tf.greater(is_first, 0), lambda: tf.gather(tags_to_ib2tags, lbl_tags)[0], lambda: tf.gather(tags_to_ib2tags, lbl_tags)[1])
#                 is_first = tf.cond(tf.greater(is_first, 0), lambda: tf.constant(0), lambda: is_first)
#                 return lbl, is_first

#             cond1 = tf.logical_and(tf.math.less(st_tags, ed_offset), tf.math.less_equal(ed_offset, ed_tags))
#             cond2 = tf.logical_and(tf.math.less_equal(st_tags, st_offset), tf.math.less(st_offset, ed_tags))
#             cond3 = tf.logical_and(tf.math.less_equal(st_offset, st_tags), tf.math.less_equal(ed_tags, ed_offset))
#             cond4 = tf.logical_or(cond1, cond2)
#             cond5 = tf.logical_or(cond4, cond3)

#             assign_idx, assign_first = tf.cond(
#                 cond5,
#                 lambda: check_first(lbl_tags, is_first),
#                 lambda: (out_idx, is_first)
#             )

#             is_first += assign_first
#             assign_idx = tf.cond(tf.greater(result2[offset_idx], 0), lambda: tf.constant(0), lambda: assign_idx)
#             result2[offset_idx] += assign_idx
            
#             return offset_idx+1, result2, offsets2, tags2, is_first

#         offset_idx = tf.constant(0)
#         is_first = tf.Variable(1, trainable=False, dtype=tf.int32)

#         offset_length = tf.shape(offsets)[0] # number of tokens
#         offset_loop_cond = lambda i, result2, offsets2, tags2, is_first: tf.less(i, offset_length) # until number of tokens
#         tgt_tag = tf.gather(tags, ne_idx)
        
#         offset_idx, result, offsets, tgt_tag, is_first = tf.while_loop(offset_loop_cond, offset_loop, [offset_idx, result, offsets, tgt_tag, is_first])
        
#         return ne_idx + 1, result, offsets, tags, out_idx

#     out_idx = tf.constant(iob2tags.index('O'))
    
#     ne_idx = tf.constant(0, dtype=tf.int32)
#     result = tf.ones_like(inputs) * out_idx

#     ne_idx, result, inputs_offsets, ne_tags, out_idx = tf.while_loop(ne_loop_cond, ne_loop, [ne_idx, result, inputs_offsets, ne_tags, out_idx])

#     x['result'] = result

#     return x

