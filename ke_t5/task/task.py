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


from ke_t5 import pipe as seq_pipe
from .task_meta import NIKL_META, KLUE_META
from . import preprocessors, metrics, postprocessors
from .utils import get_vocabulary


VOCABULARY = get_vocabulary()

GENERATIVE_OUTPUT_FEATURES = {
    "inputs": seq_pipe.Feature(
        tokenizer=VOCABULARY, add_eos=False, required=False),
    "targets": seq_pipe.Feature(
        tokenizer=VOCABULARY, add_eos=True)
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seq_pipe.Feature(
        tokenizer=VOCABULARY, add_eos=False, required=False)
}


# ================================================
# ==================== KLUE ======================
# ================================================

# ============ KLUE topic classification: Generative ============
seq_pipe.TaskRegistry.add(
    "klue_tc_gen",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'tc'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "title": "title",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='klue_tc',
            input_keys=['title'],
            label_names=KLUE_META['tc_classes'],
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('exact_match_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'tc':{
                "early_stopping": True,
                "max_length": 5,
                "num_beams": 1,
                "prefix": "klue_tc title: {}"
            },
        },
    },
    num_proc=4,
)

# ============ KLUE topic classification: Classifier ============
seq_pipe.TaskRegistry.add(
    "klue_tc",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'tc'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "title": "title",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='klue_tc',
            input_keys=['title'],
            label_names=None,
            with_feature_key=True,
            no_label_idx=7,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.accuracy_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'num_labels': len(KLUE_META['tc_classes']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['tc_classes'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['tc_classes'])},
        'problem_type': "single_label_classification",
        },
    num_proc=4,
)


# ============ KLUE NLI: Generative ============
seq_pipe.TaskRegistry.add(
    "klue_nli_gen",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'nli'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "premise": "premise",
                "hypothesis": "hypothesis",
                "label": "gold_label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='klue_nli',
            input_keys=['premise', 'hypothesis'],
            label_names=KLUE_META['nli_classes'],
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('exact_match_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'nli':{
                "early_stopping": True,
                "max_length": 5,
                "num_beams": 1,
                "prefix": "klue_nli premise: {premise} hypothesis: {hypothesis}"
            },
        },
    },
    num_proc=4,
)

# ============ KLUE NLI: Classifier ============
seq_pipe.TaskRegistry.add(
    "klue_nli",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'nli'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "premise": "premise",
                "hypothesis": "hypothesis",
                "label": "gold_label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='klue_nli',
            input_keys=['premise', 'hypothesis'],
            label_names=None,
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    metric_fns=[
        metrics.accuracy_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    output_features=DEFAULT_OUTPUT_FEATURES,
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(KLUE_META['nli_classes']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['nli_classes'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['nli_classes'])},
        'problem_type': "single_label_classification",
        },
)


# ============ KLUE STS: Generative ============
seq_pipe.TaskRegistry.add(
    "klue_sts_gen",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'sts'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence1": "sentence1",
                "sentence2": "sentence2",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_regression,
            benchmark_name='klue_sts',
            input_keys=['sentence1', 'sentence2'],
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_and_float,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.pearson_corrcoef_dict, metrics.spearman_corrcoef_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('spearman_corrcoef'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
)

# ============ KLUE STS: Regressor ============
seq_pipe.TaskRegistry.add(
    "klue_sts",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'sts'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence1": "sentence1",
                "sentence2": "sentence2",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_regression,
            benchmark_name='klue_sts',
            input_keys=['sentence1', 'sentence2'],
            is_string_tgt=False,
            with_feature_key=True
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    logit_to_id=False,
    metric_fns=[
        metrics.pearson_corrcoef_dict, metrics.spearman_corrcoef_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('spearman_corrcoef'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': 1,
        'problem_type': "regression",
        },
)


# ============ KLUE RE: Generative ============
seq_pipe.TaskRegistry.add(
    "klue_re_gen",
    seq_pipe.HFDataSource('KETI-AIR/klue', 're'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence": "sentence",
                "subject_entity": "subject_entity",
                "object_entity": "object_entity",
                "label": "label",
            }),
        functools.partial(
            preprocessors.re_preproc_for_classification,
            benchmark_name='klue_re',
            label_names=KLUE_META['re_relations']
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('exact_match_str'),
    additional_task_info={
        'task_specific_params': {
            're':{
                "early_stopping": True,
                "max_length": 10,
                "num_beams": 1,
                "prefix": "klue_re ",
            },
        },
    },
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
)

# ============ KLUE RE: Classifier ============
seq_pipe.TaskRegistry.add(
    "klue_re",
    seq_pipe.HFDataSource('KETI-AIR/klue', 're'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence": "sentence",
                "subject_entity": "subject_entity",
                "object_entity": "object_entity",
                "label": "label",
            }),
        functools.partial(
            preprocessors.re_preproc_for_classification,
            benchmark_name='klue_re',
            label_names=None
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    metric_fns=[
        metrics.accuracy_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    additional_task_info={
        'num_labels': len(KLUE_META['re_relations']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['re_relations'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['re_relations'])},
        'problem_type': "single_label_classification",
        },
)

# ============ KLUE RE: Classifier with sbj. obj. tk index ============
seq_pipe.TaskRegistry.add(
    "klue_re_tk_idx",
    seq_pipe.HFDataSource('KETI-AIR/klue', 're'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "sentence": "sentence",
                "subject_entity": "subject_entity",
                "object_entity": "object_entity",
                "label": "label",
            }),
        functools.partial(
            preprocessors.re_preproc_for_classification_with_idx,
            benchmark_name='klue_re',
            label_names=None
        ),
        functools.partial(
            preprocessors.tokenize_re_with_tk_idx,
            output_features=DEFAULT_OUTPUT_FEATURES,
            input_key='inputs'
        ),
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "label",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    model_input_columns=['input_ids', 'attention_mask', 'labels', 'entity_token_idx'],
    num_proc=4,
    metric_fns=[
        metrics.accuracy_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    additional_task_info={
        'num_labels': len(KLUE_META['re_relations']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['re_relations'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['re_relations'])},
        'problem_type': "single_label_classification",
        },
)

# ============ KLUE MRC: Generative ============
seq_pipe.TaskRegistry.add(
    "klue_mrc_gen",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'mrc'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "context": "context",
                "question": "question",
                "answers": "answers",
                "is_impossible": "is_impossible",
            }),
        functools.partial(
            preprocessors.preprocess_quad,
            benchmark_name='klue_mrc',
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict, metrics.f1_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('f1_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'mrc':{
                "early_stopping": True,
                "max_length": 30,
                "num_beams": 4,
                "prefix": "klue_mrc question: {question} context: {context}",
            },
        },
    },
    num_proc=4,
)

# ============ KLUE MRC: Generative - Context free ============
seq_pipe.TaskRegistry.add(
    "klue_mrc_gen_context_free",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'mrc'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "context": "context",
                "question": "question",
                "answers": "answers",
                "is_impossible": "is_impossible",
            }),
        functools.partial(
            preprocessors.preprocess_quad,
            benchmark_name='klue_mrc',
            include_context=False
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict, metrics.f1_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('f1_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'mrc':{
                "early_stopping": True,
                "max_length": 30,
                "num_beams": 4,
                "prefix": "klue_mrc trivia question: {question}",
            },
        },
    },
    num_proc=4,
)


# ============ KLUE NER: Classifier ============
seq_pipe.TaskRegistry.add(
    "klue_ner",
    seq_pipe.HFDataSource('KETI-AIR/klue', 'ner'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "guid",
                "inputs": "text",
                "tag_info": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=KLUE_META['ner_tags'],
            iob2_tags=KLUE_META[ 'ner_iob2_tags'],
            tag_label='tag_info'
        ),
        functools.partial(
            seq_pipe.preprocessors.trim_and_pad,
            key_pad_id_map={
                "inputs": VOCABULARY.pad_token_id,
                "targets": KLUE_META[ 'ner_iob2_tags'].index('O'),
            }
        ),
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.token_accuracy_dict_variable_length, 
        functools.partial(metrics.char_level_f1_score_klue_dict, iob2_tags=KLUE_META['ner_iob2_tags'])
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('token_accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'num_labels': len(KLUE_META['ner_iob2_tags']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['ner_iob2_tags'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['ner_iob2_tags'])},
        },
    num_proc=4,
)



# ================================================
# ==================== NIKL ======================
# ================================================

# ============ NIKL NER: Classifier ============
seq_pipe.TaskRegistry.add(
    "nikl_ner",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'ne.v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "inputs": "form",
                "tag_info": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne_tags'],
            iob2_tags=NIKL_META['ne_iob2_tags'],
            tag_label='tag_info'
        ),
        functools.partial(
            seq_pipe.preprocessors.trim_and_pad,
            key_pad_id_map={
                "inputs": VOCABULARY.pad_token_id,
                "targets": NIKL_META['ne_iob2_tags'].index('O'),
            }
        ),
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.token_accuracy_dict_variable_length
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('token_accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne_iob2_tags']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne_iob2_tags'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne_iob2_tags'])},
        },
)

# ============ NIKL NER: Classifier - split ============
seq_pipe.TaskRegistry.add(
    "nikl_ner_split",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'ne.v1.0.split'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "inputs": "form",
                "tag_info": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne_tags'],
            iob2_tags=NIKL_META['ne_iob2_tags'],
            tag_label='tag_info'
        ),
        functools.partial(
            seq_pipe.preprocessors.trim_and_pad,
            key_pad_id_map={
                "inputs": VOCABULARY.pad_token_id,
                "targets": NIKL_META['ne_iob2_tags'].index('O'),
            }
        ),
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.token_accuracy_dict_variable_length
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('token_accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne_iob2_tags']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne_iob2_tags'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne_iob2_tags'])},
        },
)


# ============ NIKL NER2020: Classifier ============
seq_pipe.TaskRegistry.add(
    "nikl_ner2020",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'ne.2020.v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "inputs": "form",
                "tag_info": "ne",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne2020_tags'],
            iob2_tags=NIKL_META['ne2020_iob2_tags'],
            tag_label='tag_info'
        ),
        functools.partial(
            seq_pipe.preprocessors.trim_and_pad,
            key_pad_id_map={
                "inputs": VOCABULARY.pad_token_id,
                "targets": NIKL_META['ne2020_iob2_tags'].index('O'),
            }
        ),
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.token_accuracy_dict_variable_length
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('token_accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne2020_iob2_tags']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne2020_iob2_tags'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne2020_iob2_tags'])},
        },
)

# ============ NIKL NER2020: Classifier - split ============
seq_pipe.TaskRegistry.add(
    "nikl_ner2020_split",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'ne.2020.v1.0.split'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "inputs": "form",
                "tag_info": "ne",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne2020_tags'],
            iob2_tags=NIKL_META['ne2020_iob2_tags'],
            tag_label='tag_info'
        ),
        functools.partial(
            seq_pipe.preprocessors.trim_and_pad,
            key_pad_id_map={
                "inputs": VOCABULARY.pad_token_id,
                "targets": NIKL_META['ne2020_iob2_tags'].index('O'),
            }
        ),
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.token_accuracy_dict_variable_length
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('token_accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne2020_iob2_tags']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne2020_iob2_tags'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne2020_iob2_tags'])},
        },
)


# ============ NIKL summarization summary: Generative ============
seq_pipe.TaskRegistry.add(
    "nikl_summarization_summary",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'summarization.v1.0.summary'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "document_id",
                "inputs": "article",
                "targets": "highlights",
            }),
        functools.partial(
            preprocessors.base_preproc_for_conditional_generation,
            prefix='summarize_summary:',
            input_keys=['inputs'],
            with_feature_key=False,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.bleu_dict, metrics.rouge_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('rougeLsum'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'summarization':{
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize_summary: "
            },
        },
    },
    num_proc=4,
)

# ============ NIKL summarization summary: Generative - split ============
seq_pipe.TaskRegistry.add(
    "nikl_summarization_summary_split",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'summarization.v1.0.summary.split'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "document_id",
                "inputs": "article",
                "targets": "highlights",
            }),
        functools.partial(
            preprocessors.base_preproc_for_conditional_generation,
            prefix='summarize_summary:',
            input_keys=['inputs'],
            with_feature_key=False,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.bleu_dict, metrics.rouge_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('rougeLsum'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'summarization':{
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize_summary: "
            },
        },
    },
    num_proc=4,
)


# ============ NIKL summarization topic: Generative ============
seq_pipe.TaskRegistry.add(
    "nikl_summarization_topic",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'summarization.v1.0.topic'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "document_id",
                "inputs": "article",
                "targets": "highlights",
            }),
        functools.partial(
            preprocessors.base_preproc_for_conditional_generation,
            prefix='summarize_topic:',
            input_keys=['inputs'],
            with_feature_key=False,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    metric_fns=[
        metrics.bleu_dict, metrics.rouge_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('rougeLsum'),
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'summarization':{
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize_topic: "
            },
        },
    },
    num_proc=4,
)

# ============ NIKL summarization topic: Generative - split ============
seq_pipe.TaskRegistry.add(
    "nikl_summarization_topic_split",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'summarization.v1.0.topic.split'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "document_id",
                "inputs": "article",
                "targets": "highlights",
            }),
        functools.partial(
            preprocessors.base_preproc_for_conditional_generation,
            prefix='summarize_topic:',
            input_keys=['inputs'],
            with_feature_key=False,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    metric_fns=[
        metrics.bleu_dict, metrics.rouge_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('rougeLsum'),
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'summarization':{
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize_topic: "
            },
        },
    },
    num_proc=4,
)

# ============ NIKL CoLA: Generative ============
seq_pipe.TaskRegistry.add(
    "nikl_cola_gen",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'cola.v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "idx",
                "sentence": "sentence",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='nikl_cola',
            input_keys=['sentence'],
            label_names=NIKL_META['cola_classes'],
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_and_string_to_label,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY,
            label_info=NIKL_META['cola_classes'],
        ),
    metric_fns=[
        metrics.matthews_corrcoef_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('matthews_corrcoef'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'cola':{
                "early_stopping": True,
                "max_length": 5,
                "num_beams": 1,
                "prefix": "nikl_cola sentence: {}"
            },
        },
    },
    num_proc=4,
)


# ============ NIKL CoLA: Classifier ============
seq_pipe.TaskRegistry.add(
    "nikl_cola",
    seq_pipe.HFDataSource('KETI-AIR/nikl', 'cola.v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "idx",
                "sentence": "sentence",
                "label": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='nikl_cola',
            input_keys=['sentence'],
            label_names=None,
            with_feature_key=True,
            no_label_idx=0,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.matthews_corrcoef_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('matthews_corrcoef'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'num_labels': len(NIKL_META['cola_classes']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['cola_classes'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['cola_classes'])},
        'problem_type': "single_label_classification",
        },
    num_proc=4,
)


# ================================================
# =================== KorQuAD ====================
# ================================================


# ============ KorQuAD v1.0: Generative ============
seq_pipe.TaskRegistry.add(
    "korquad_gen",
    seq_pipe.HFDataSource('KETI-AIR/korquad', 'v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "context": "context",
                "question": "question",
                "answers": "answers",
            }),
        functools.partial(
            preprocessors.preprocess_quad,
            benchmark_name='korquad',
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict, metrics.f1_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('f1_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'mrc':{
                "early_stopping": True,
                "max_length": 30,
                "num_beams": 4,
                "prefix": "korquad question: {question} context: {context}",
            },
        },
    },
    num_proc=4,
)

# ============ KorQuAD v1.0: Generative - Context free ============
seq_pipe.TaskRegistry.add(
    "korquad_gen_context_free",
    seq_pipe.HFDataSource('KETI-AIR/korquad', 'v1.0'),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "id": "id",
                "context": "context",
                "question": "question",
                "answers": "answers",
            }),
        functools.partial(
            preprocessors.preprocess_quad,
            benchmark_name='korquad',
            include_context=False
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict, metrics.f1_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('f1_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'mrc':{
                "early_stopping": True,
                "max_length": 30,
                "num_beams": 4,
                "prefix": "korquad trivia question: {question}",
            },
        },
    },
    num_proc=4,
)



# ================================================
# ================== Kor 3i4k ====================
# ================================================
_KOR_3I4K_CLASSES = ["fragment", "statement", "question", "command", "rhetorical question", "rhetorical command", "intonation-depedent utterance"]
# ============ Kor 3i4k classification: Generative ============
seq_pipe.TaskRegistry.add(
    "kor_3i4k_gen",
    seq_pipe.HFDataSource('kor_3i4k', None),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "text": "text",
                "targets": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='kor_3i4k',
            input_keys=['text'],
            label_names=_KOR_3I4K_CLASSES,
            with_feature_key=True,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.decode_for_generator,
            decode_keys=['predictions', 'labels'],
            tokenizer=VOCABULARY
        ),
    metric_fns=[
        metrics.exact_match_str_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('exact_match_str'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'task_specific_params': {
            'kor_3i4k':{
                "early_stopping": True,
                "max_length": 5,
                "num_beams": 1,
                "prefix": "kor_3i4k text: {}"
            },
        },
    },
    num_proc=4,
)

# ============ Kor 3i4k: Classifier ============
seq_pipe.TaskRegistry.add(
    "kor_3i4k",
    seq_pipe.HFDataSource('kor_3i4k', None),
    preprocessors=[
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "text": "text",
                "targets": "label",
            }),
        functools.partial(
            preprocessors.base_preproc_for_classification,
            benchmark_name='kor_3i4k',
            input_keys=['text'],
            label_names=None,
            with_feature_key=True,
            no_label_idx=0,
        ),
        seq_pipe.preprocessors.tokenize_output_features, 
        seq_pipe.preprocessors.append_eos_after_trim_output_features,
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
                "labels": "targets",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[
        metrics.accuracy_dict
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    model_input_columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'num_labels': len(_KOR_3I4K_CLASSES),
        'id2label': {idx:key for idx, key in enumerate(_KOR_3I4K_CLASSES)},
        'label2id': {key:idx for idx, key in enumerate(_KOR_3I4K_CLASSES)},
        'problem_type': "single_label_classification",
        },
    num_proc=4,
)





if __name__ == "__main__":
    seq_pipe.set_hf_data_dir_override("./data")
    seq_pipe.set_hf_cache_dir_override("./cache_dir/huggingface_datasets")

    task = seq_pipe.get_task('klue_re_tk_idx')
    
    dataset = task.get_dataset(
        sequence_length={"inputs": 512, "targets": 512},
        split="train"
    )
    

    # Print the first 5 examples.
    for _, ex in zip(range(5), iter(dataset)):
        print(ex)

    # import torch

    # dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # print(next(iter(dataloader)))
    
