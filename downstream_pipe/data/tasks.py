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

import functools

import seqio
import tensorflow as tf

from .task_meta import NIKL_META, KLUE_META
from .preprocessors import (
    base_preproc_for_classification, 
    base_preproc_for_regression, 
    re_preproc_for_classification,
    preprocess_quad,
    tokenize_with_offsets)
from .utils import get_vocabulary

MAX_NUM_CLASSES=100

VOCABULARY = get_vocabulary()

GENERATIVE_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=VOCABULARY, add_eos=False, required=False, dtype=tf.int32),
    "targets": seqio.Feature(
        vocabulary=VOCABULARY, add_eos=True, dtype=tf.int32)
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=VOCABULARY, add_eos=False, required=False, dtype=tf.int32)
}

# ============ KLUE topic classification: Generative ============
seqio.TaskRegistry.add(
    "klue_tc_gen",
    seqio.TfdsDataSource(tfds_name="klue/tc:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "title": "title",
                "label": "label",
            }),
        functools.partial(
            base_preproc_for_classification,
            benchmark_name='klue_tc',
            input_keys=['title'],
            label_names=KLUE_META['tc_classes'],
            with_feature_key=True,
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE topic classification: Classifier ============
seqio.TaskRegistry.add(
    "klue_tc",
    seqio.TfdsDataSource(tfds_name="klue/tc:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "title": "title",
                "label": "label",
            }),
        functools.partial(
            base_preproc_for_classification,
            benchmark_name='klue_tc',
            input_keys=['title'],
            label_names=None,
            with_feature_key=True,
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

# ============ KLUE NLI: Generative ============
seqio.TaskRegistry.add(
    "klue_nli_gen",
    seqio.TfdsDataSource(tfds_name="klue/nli:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "premise": "premise",
                "hypothesis": "hypothesis",
                "label": "gold_label",
            }),
        functools.partial(
            base_preproc_for_classification,
            benchmark_name='klue_nli',
            input_keys=['premise', 'hypothesis'],
            label_names=KLUE_META['nli_classes'],
            with_feature_key=True,
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE NLI: Classifier ============
seqio.TaskRegistry.add(
    "klue_nli",
    seqio.TfdsDataSource(tfds_name="klue/nli:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "premise": "premise",
                "hypothesis": "hypothesis",
                "label": "gold_label",
            }),
        functools.partial(
            base_preproc_for_classification,
            benchmark_name='klue_nli',
            input_keys=['premise', 'hypothesis'],
            label_names=None,
            with_feature_key=True,
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

# ============ KLUE STS: Generative ============
seqio.TaskRegistry.add(
    "klue_sts_gen",
    seqio.TfdsDataSource(tfds_name="klue/sts:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence1": "sentence1",
                "sentence2": "sentence2",
                "label": "label",
            }),
        functools.partial(
            base_preproc_for_regression,
            benchmark_name='klue_sts',
            input_keys=['sentence1', 'sentence2'],
            with_feature_key=True,
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE STS: Regressor ============
seqio.TaskRegistry.add(
    "klue_sts",
    seqio.TfdsDataSource(tfds_name="klue/sts:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence1": "sentence1",
                "sentence2": "sentence2",
                "label": "label",
            }),
        functools.partial(
            base_preproc_for_regression,
            benchmark_name='klue_sts',
            input_keys=['sentence1', 'sentence2'],
            is_string_tgt=False,
            with_feature_key=True
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
)


# ============ KLUE RE: Generative ============
seqio.TaskRegistry.add(
    "klue_re_gen",
    seqio.TfdsDataSource(tfds_name="klue/re:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence": "sentence",
                "subject_entity": "subject_entity",
                "object_entity": "object_entity",
                "label": "label",
            }),
        functools.partial(
            re_preproc_for_classification,
            benchmark_name='klue_re',
            label_names=KLUE_META['re_relations']
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE RE: Classifier ============
seqio.TaskRegistry.add(
    "klue_re",
    seqio.TfdsDataSource(tfds_name="klue/re:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence": "sentence",
                "subject_entity": "subject_entity",
                "object_entity": "object_entity",
                "label": "label",
            }),
        functools.partial(
            re_preproc_for_classification,
            benchmark_name='klue_re',
            label_names=None
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
)

# ============ KLUE MRC: Generative ============
seqio.TaskRegistry.add(
    "klue_mrc_gen",
    seqio.TfdsDataSource(tfds_name="klue/mrc:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "context": "context",
                "question": "question",
                "answers": "answers",
                "is_impossible": "is_impossible",
            }),
        functools.partial(
            preprocess_quad,
            benchmark_name='klue_mrc',
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE MRC: Generative - Context free ============
seqio.TaskRegistry.add(
    "klue_mrc_gen_context_free",
    seqio.TfdsDataSource(tfds_name="klue/mrc:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "context": "context",
                "question": "question",
                "answers": "answers",
                "is_impossible": "is_impossible",
            }),
        functools.partial(
            preprocess_quad,
            benchmark_name='klue_mrc',
            include_context=False
        ),
        seqio.preprocessors.tokenize, 
        seqio.preprocessors.append_eos
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
)

# ============ KLUE MRC: Token Classification ============
seqio.TaskRegistry.add(
    "klue_ner",
    seqio.TfdsDataSource(tfds_name="klue/ner:1.0.0"),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "inputs": "text",
                "NE": "NE",
            }),
        tokenize_with_offsets,
        functools.partial(
            preprocess_iob2, 
            tag_key="inputs",
            tags=KLUE_META["ner_tags"],
            iob2tags=KLUE_META["ner_iob2_tags"],
            ),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
)
