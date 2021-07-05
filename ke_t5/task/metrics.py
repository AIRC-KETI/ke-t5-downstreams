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

from absl import logging
from collections import Counter
import numpy as np
from numpy.lib.function_base import average
import sacrebleu
import scipy.stats
import sklearn.metrics
from rouge_score import rouge_scorer
from rouge_score import scoring


def accuracy_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    return {"accuracy": {'score': 100*sklearn.metrics.accuracy_score(targets, predictions), 'count': len(targets)}}


def f1_score_dict_micro_sample_weight(gathered_dict, target_key='labels', prediction_key='predictions'):
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    # At Klue RE, micro F1 Scores exclude no_relation(0) 
    sample_weight = np.array(targets != 0, dtype=np.float)
    return {"micro_F1_sample_weight ": {'score': 100*sklearn.metrics.f1_score(targets, predictions, average='micro', sample_weight=sample_weight, zero_division = 0), 'count': np.count_nonzero(targets)}}


def bleu_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    """Computes BLEU score.
    Args:
      targets: list of strings or list of list of strings if multiple references
        are present.
      predictions: list of strings
    Returns:
      bleu_score across all targets and predictions
    """
    _targets = gathered_dict[target_key]
    targets = _targets
    predictions = gathered_dict[prediction_key]
    if isinstance(targets[0], list):
        targets = [[x for x in target] for target in targets]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        targets = [targets]

    bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                       smooth_method="exp",
                                       smooth_value=0.0,
                                       force=False,
                                       lowercase=False,
                                       tokenize="intl",
                                       use_effective_order=False)
    return {"bleu": {'score': bleu_score.score, 'count': len(_targets)}}


def rouge_dict(gathered_dict, target_key='labels', prediction_key='predictions', score_keys=None):
    """Computes rouge score.
    Args:
      targets: list of strings
      predictions: list of strings
      score_keys: list of strings with the keys to compute.
    Returns:
      dict with score_key: rouge score across all targets and predictions
    """
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]

    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)
    aggregator = scoring.BootstrapAggregator()

    def _prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(" . ", " .\n")
        return summary

    for prediction, target in zip(predictions, targets):
        target = _prepare_summary(target)
        prediction = _prepare_summary(prediction)
        aggregator.add_scores(scorer.score(
            target=target, prediction=prediction))
    result = aggregator.aggregate()
    return {key: {'score': result[key].mid.fmeasure*100, 'count': len(targets)} for key in score_keys}


def exact_match_str_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    return {"exact_match_str": {'score': 100 * np.average(np.array([x==y for x, y in zip(targets, predictions)], dtype=np.float)), 'count': len(targets)}}

def f1_str_base(target, prediction):
    target = [ch for ch in target]
    prediction = [ch for ch in prediction]

    same = Counter(target) & Counter(prediction)
    num_same = sum(same.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(target)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def f1_str_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    return {"f1_str": {'score': 100 * np.average(np.array([f1_str_base(x, y) for x, y in zip(targets, predictions)], dtype=np.float)), 'count': len(targets)}}


def pearson_corrcoef_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    """Pearson correlation coefficient."""
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    predictions = np.squeeze(predictions)
    return {"pearson_corrcoef":
            {'score': 100 * scipy.stats.pearsonr(targets, predictions)[0], 'count': len(targets)}}


def spearman_corrcoef_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    """Spearman correlation coefficient."""
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    predictions = np.squeeze(predictions)
    return {"spearman_corrcoef":
            {'score': 100 * scipy.stats.spearmanr(targets, predictions)[0], 'count': len(targets)}}


def token_accuracy_dict(gathered_dict, target_key='labels', target_weights_key='targets_attention_mask', prediction_key='predictions'):
    """Spearman correlation coefficient."""
    targets = gathered_dict[target_key].flatten()
    predictions = gathered_dict[prediction_key].flatten()
    target_weights = gathered_dict[target_weights_key] if target_weights_key in gathered_dict else None
    if target_weights is not None:
        target_weights = target_weights.flatten()
        return {"token_accuracy": {'score': 100*sklearn.metrics.accuracy_score(targets, predictions, sample_weight=target_weights), 'count': np.sum(target_weights)}}
    else:
        return {"token_accuracy": {'score': 100*sklearn.metrics.accuracy_score(targets, predictions), 'count': len(targets)}}

def matthews_corrcoef_dict(gathered_dict, target_key='labels', prediction_key='predictions'):
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]
    return {"matthews_corrcoef": {'score': sklearn.metrics.matthews_corrcoef(targets, predictions), 'count': len(targets)}}


def token_accuracy_dict_variable_length(gathered_dict, target_key='labels', prediction_key='predictions', **unused_kwargs):
    """Spearman correlation coefficient."""
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]

    _cnt = 0
    _sum = 0

    for idx, target in enumerate(targets):
        prediction = predictions[idx]
        for t, p in zip(target, prediction):
            if t == p:
                _sum += 1
            _cnt += 1
    
    return {"token_accuracy": {'score': 100*_sum/_cnt, 'count': _cnt}}


def create_char_tags(prediction, char_to_token, char_tag, iob2_tags):
    char_tag_pred = [iob2_tags.index('O')] * len(char_tag)
    
    for pred_token_idx, pred_label in enumerate(prediction):
        tag_txt = iob2_tags[pred_label]
        if tag_txt != 'O':
            if tag_txt[0] == 'B':
                beg_tk_id = pred_label
                end_tk_id = iob2_tags.index('I'+tag_txt[1:])
                is_first=True
                for ch_idx, token_idx in enumerate(char_to_token):
                    if token_idx == pred_token_idx:
                        if is_first:
                            char_tag_pred[ch_idx] = beg_tk_id
                            is_first=False
                        else:
                            char_tag_pred[ch_idx] = end_tk_id
            else:
                for ch_idx, token_idx in enumerate(char_to_token):
                    if token_idx == pred_token_idx:
                        char_tag_pred[ch_idx] = pred_label
    return char_tag_pred

def char_level_f1_score_klue_dict(gathered_dict, iob2_tags, target_key='labels', prediction_key='predictions', klue_metric_key='klue_metric', **unused_kwargs):
    """Spearman correlation coefficient."""
    targets = gathered_dict[target_key]
    predictions = gathered_dict[prediction_key]

    klue_metric_info = gathered_dict[klue_metric_key]
    char_to_token = klue_metric_info['char_to_token']
    char_tag = klue_metric_info['char_tag']

    _cnt = 0
    _sum = 0

    for target, prediction, ch2tk, chlvl_tag in zip(targets, predictions, char_to_token, char_tag):
        chlvl_tag_pred = create_char_tags(prediction, ch2tk, chlvl_tag, iob2_tags)

        chlvl_tag_np = np.array(chlvl_tag)
        chlvl_tag_pred_np = np.array(chlvl_tag_pred)
        sample_weight = np.array(chlvl_tag_np != iob2_tags.index('O'), dtype=np.float)

        sum_sample_w = np.sum(sample_weight)

        if sum_sample_w > 0:
            _sum += 100*sklearn.metrics.f1_score(chlvl_tag_np, chlvl_tag_pred_np, average='macro', sample_weight=sample_weight)
            _cnt += int(sum_sample_w)
    
    return {"char_f1": {'score': _sum/_cnt, 'count': _cnt}}






# # adopted from 't5' github
# def accuracy(targets, predictions):
#     return {"accuracy": 100*sklearn.metrics.accuracy_score(targets, predictions)}



# # adopted from 't5' github
# def bleu(targets, predictions):
#     """Computes BLEU score.
#     Args:
#       targets: list of strings or list of list of strings if multiple references
#         are present.
#       predictions: list of strings
#     Returns:
#       bleu_score across all targets and predictions
#     """
#     if isinstance(targets[0], list):
#         targets = [[x for x in target] for target in targets]
#     else:
#         # Need to wrap targets in another list for corpus_bleu.
#         targets = [targets]

#     bleu_score = sacrebleu.corpus_bleu(predictions, targets,
#                                        smooth_method="exp",
#                                        smooth_value=0.0,
#                                        force=False,
#                                        lowercase=False,
#                                        tokenize="intl",
#                                        use_effective_order=False)
#     return {"bleu": bleu_score.score}

# # adopted from 't5' github
# def rouge(targets, predictions, score_keys=None):
#     """Computes rouge score.
#     Args:
#       targets: list of strings
#       predictions: list of strings
#       score_keys: list of strings with the keys to compute.
#     Returns:
#       dict with score_key: rouge score across all targets and predictions
#     """

#     if score_keys is None:
#         score_keys = ["rouge1", "rouge2", "rougeLsum"]
#     scorer = rouge_scorer.RougeScorer(score_keys)
#     aggregator = scoring.BootstrapAggregator()

#     def _prepare_summary(summary):
#         # Make sure the summary is not bytes-type
#         # Add newlines between sentences so that rougeLsum is computed correctly.
#         summary = summary.replace(" . ", " .\n")
#         return summary

#     for prediction, target in zip(predictions, targets):
#         target = _prepare_summary(target)
#         prediction = _prepare_summary(prediction)
#         aggregator.add_scores(scorer.score(
#             target=target, prediction=prediction))
#     result = aggregator.aggregate()
#     # for key in score_keys:
#     #     logging.info(
#     #         "%s = %.2f, 95%% confidence [%.2f, %.2f]",
#     #         key,
#     #         result[key].mid.fmeasure*100,
#     #         result[key].low.fmeasure*100,
#     #         result[key].high.fmeasure*100,
#     #     )
#     return {key: result[key].mid.fmeasure*100 for key in score_keys}



# # adopted from 't5' github
# def pearson_corrcoef(targets, predictions):
#     """Pearson correlation coefficient."""
#     return {"pearson_corrcoef":
#             100 * scipy.stats.pearsonr(targets, predictions)[0]}

# # adopted from 't5' github
# def spearman_corrcoef(targets, predictions):
#     """Spearman correlation coefficient."""
#     return {"spearman_corrcoef":
#             100 * scipy.stats.spearmanr(targets, predictions)[0]}

# # adopted from 't5' github
# def exact_match(targets, predictions):
#     """Computes whether the targets match predictions exactly."""
#     return {"exact_match": 100 * float(np.array_equal(targets, predictions))}


# def exact_match_str(target, prediction):
#     return {"exact_match_str": 1. if target == prediction else 0.}


# def f1_str_base(target, prediction):
#     target = [ch for ch in target]
#     prediction = [ch for ch in prediction]

#     same = Counter(target) & Counter(prediction)
#     num_same = same.values()
#     if num_same == 0:
#         return 0
    
#     precision = 1.0 * num_same / len(prediction)
#     recall = 1.0 * num_same / len(target)
#     f1 = (2 * precision * recall) / (precision + recall)
    
#     return f1

# def f1_str(target, prediction):
#     return {"f1_str": 100 * f1_str_base(target, prediction)}

# def f1_str_batch(targets, predictions):
#     return {"f1_str": 100 * np.average(np.array([f1_str_base(x, y) for x, y in zip(targets, predictions)], dtype=np.float))}
# sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)[source]
