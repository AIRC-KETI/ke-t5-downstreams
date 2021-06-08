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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    train_metric_fns=[
        metrics.exact_match_str_batch
    ],
    columns=['input_ids', 'attention_mask', 'labels'],
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
    train_metric_fns=[
        metrics.accuracy
    ],
    best_fn=seq_pipe.evaluation.GreaterIsTheBest('accuracy'),
    columns=['input_ids', 'attention_mask', 'labels'],
    additional_task_info={
        'num_labels': len(KLUE_META['tc_classes']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['tc_classes'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['tc_classes'])},
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    train_metric_fns=[
        metrics.exact_match_str_batch
    ],
    columns=['input_ids', 'attention_mask', 'labels'],
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
    train_metric_fns=[
        metrics.accuracy
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(KLUE_META['nli_classes']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['nli_classes'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['nli_classes'])},
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
    columns=['input_ids', 'attention_mask', 'labels'],
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
    columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    train_metric_fns=[
        metrics.exact_match_str_batch
    ],
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
    columns=['input_ids', 'attention_mask', 'labels'],
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
    columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    train_metric_fns=[
        metrics.accuracy
    ],
    additional_task_info={
        'num_labels': len(KLUE_META['re_relations']),
        'id2label': {idx:key for idx, key in enumerate(KLUE_META['re_relations'])},
        'label2id': {key:idx for idx, key in enumerate(KLUE_META['re_relations'])},
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    train_metric_fns=[
        metrics.exact_match_str_batch
    ],
    columns=['input_ids', 'attention_mask', 'labels'],
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    train_metric_fns=[
        metrics.exact_match_str_batch
    ],
    columns=['input_ids', 'attention_mask', 'labels'],
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
                "NE": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=KLUE_META['ner_tags'],
            iob2_tags=KLUE_META['ner_iob2_tags'],
            tag_label='NE'
        ),
        seq_pipe.preprocessors.trim_and_pad_output_features,
        functools.partial(
            seq_pipe.preprocessors.rekey, 
            key_map={
                "input_ids": "inputs",
                "attention_mask": "inputs_attention_mask",
            }),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
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
                "NE": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne_tags'],
            iob2_tags=NIKL_META['ne_iob2_tags'],
            tag_label='NE'
        ),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    columns=['input_ids', 'attention_mask', 'labels'],
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
                "NE": "NE",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne_tags'],
            iob2_tags=NIKL_META['ne_iob2_tags'],
            tag_label='NE'
        ),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    columns=['input_ids', 'attention_mask', 'labels'],
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
                "NE": "ne",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne2020_tags'],
            iob2_tags=NIKL_META['ne2020_iob2_tabs'],
            tag_label='NE'
        ),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne2020_iob2_tabs']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne2020_iob2_tabs'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne2020_iob2_tabs'])},
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
                "NE": "ne",
            }),
        functools.partial(
            preprocessors.tokenize_and_preproc_iob2,
            tags=NIKL_META['ne2020_tags'],
            iob2_tags=NIKL_META['ne2020_iob2_tabs'],
            tag_label='NE'
        ),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    columns=['input_ids', 'attention_mask', 'labels'],
    num_proc=4,
    additional_task_info={
        'num_labels': len(NIKL_META['ne2020_iob2_tabs']),
        'id2label': {idx:key for idx, key in enumerate(NIKL_META['ne2020_iob2_tabs'])},
        'label2id': {key:idx for idx, key in enumerate(NIKL_META['ne2020_iob2_tabs'])},
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    columns=['input_ids', 'attention_mask', 'labels'],
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
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    columns=['input_ids', 'attention_mask', 'labels'],
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
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    columns=['input_ids', 'attention_mask', 'labels'],
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
    output_features=GENERATIVE_OUTPUT_FEATURES,
    train_postprocess_fn=functools.partial(
            postprocessors.postprocess_for_generator,
            tokenizer=VOCABULARY
        ),
    columns=['input_ids', 'attention_mask', 'labels'],
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


if __name__ == "__main__":
    seq_pipe.set_hf_data_dir_override("./data")
    seq_pipe.set_hf_cache_dir_override("./cache_dir/huggingface_datasets")

    task = seq_pipe.get_task('nikl_ner2020_split')
    
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
    
