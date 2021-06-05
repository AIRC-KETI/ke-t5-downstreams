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
        pattern = pattern_tmpl.replace('N', str(span_idx))
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
