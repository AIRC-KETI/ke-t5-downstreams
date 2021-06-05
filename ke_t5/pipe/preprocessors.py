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

from typing import Mapping, Optional, Any

import datasets

from . import utils
from . import dataset_providers

OutputFeaturesType = Mapping[str, dataset_providers.Feature]
SequenceLengthType = Mapping[str, int]


@utils.map_over_dataset
def print_dataset(features):
    return {k: print(v, [v], k + ': ') for k, v in features.items()}


# adopted from 'seqio' github
@utils.map_over_dataset
def rekey(x, key_map=None):
    """Replace the feature keys according to the mapping in `key_map`.

    For example, if the dataset returns examples of the format:
    {'foo': 'something', 'bar': 'something else'}
    and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
    examples with the format
    {'boo': 'something', 'spar': 'something else'}

    If a mapping is to an empty key or None, set the new key to an empty string.
    Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
    Returns:
      A preprocessed example with the format listed above.
    """
    if key_map:
        return {
            new_key: x[old_key] if old_key else ''
            for new_key, old_key in key_map.items()
        }
    return x

@utils.map_over_dataset
def rename_key(x, key_map=None, delete_old_key=True):
    if key_map:
        for new_key, old_key in key_map.items():
            if old_key in x:
                x[new_key] = x[old_key]
                if delete_old_key:
                    del x[old_key]
    return x


@utils.map_over_dataset
def trim_output_features(
    example,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType]
) -> datasets.Dataset:
    """Trim output features to sequence length."""
    def _trim(k: str, v) -> Any:
        if k not in output_features or not sequence_length:
            return v
        return v[:sequence_length[k]]

    return {k: _trim(k, v) for k, v in example.items()}


@utils.map_over_dataset
def tokenize_output_features(
    features,
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
    with_eos: bool = False
) -> datasets.Dataset:
    """Encode output features with specified vocbularies.

    Passes through other features unchanged. Optionally passes through copy
    of original features with "_pretokenized" suffix added to the key.

    Args:
      dataset: a datasets.Dataset of examples to tokenize.
      output_features: a dict of Feature objects; their vocabulary attribute will
        be used to tokenize the specified features.
      copy_pretokenized: bool, whether to pass through copies of original features
        with "_pretokenized" suffix added to the key.
      with_eos: bool, whether to append EOS to the end of the sequence.

    Returns:
      a tf.data.Dataset
    """
    ret = {}
    for k, v in features.items():
        if k in output_features:
            if copy_pretokenized:
                ret[f'{k}_pretokenized'] = v
            tokenizer = output_features[k].tokenizer
            v = tokenizer(v).input_ids
            eos_token_id = tokenizer.eos_token_id
            if with_eos and output_features[k].add_eos:
                if eos_token_id is None:
                    raise AttributeError('tokenizer must have eos_token to add eos id but eos token is : {}'.format(
                        eos_token_id))
                if v[-1] != eos_token_id:
                    v = v + [eos_token_id]
            elif eos_token_id is not None:
                if v[-1] == eos_token_id:
                    v = v[:-1]
        ret[k] = v
    return ret


def tokenize_and_append_eos_output_features(
    dataset: datasets.Dataset,
    output_features: OutputFeaturesType,
    copy_pretokenized: bool = True,
) -> datasets.Dataset:
    """Encode output features with specified vocbularies and append EOS.

    Passes through non-string features unchanged. Optionally passes through copy
    of original features with "_pretokenized" suffix added to the key.

    Args:
      dataset: a datasets.Dataset of examples to tokenize.
      output_features: a dict of Feature objects; their vocabulary attribute will
        be used to tokenize the specified features.
      copy_pretokenized: bool, whether to pass through copies of original features
        with "_pretokenized" suffix added to the key.

    Returns:
      a datasets.Dataset
    """
    return tokenize_output_features(dataset, output_features, copy_pretokenized, with_eos=True)


@utils.map_over_dataset
def append_eos_output_features(
    features,
    output_features: OutputFeaturesType,
) -> datasets.Dataset:
    """Appends EOS to output feature token sequences with `add_eos` set to True.

    Args:
      dataset: a tf.data.Dataset of tokenized examples to preprocess.
      output_features: a mapping of output feature names to Feature objects.

    Returns:
      a datasets.Dataset of tokenized examples with EOS added to specified output
      features.
    """
    ret = {}
    for k, v in features.items():
        if k in output_features:
            tokenizer = output_features[k].tokenizer
            eos_token_id = tokenizer.eos_token_id
            if output_features[k].add_eos:
                if eos_token_id is None:
                    raise AttributeError('tokenizer must have eos_token to add eos id but eos token is : {}'.format(
                        eos_token_id))
                if v[-1] != eos_token_id:
                    v = v + [eos_token_id]
        ret[k] = v
    return ret


@utils.map_over_dataset
def append_eos_after_trim_output_features(
    features,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None,
) -> datasets.Dataset:
    """Trims output feature token sequences and then appends EOS.

    Respects the `add_eos` field of the seqio.Features in `output_features`.
    Truncates features before adding the EOS to ensure they fit in the max length
    specified by `sequence_length` once the EOS is added. If `sequence_length` is
    None, no trimming is performed.

    Note that sequences are automatically trimmed at the end of the Task pipeline,
    so unless you want the features to always end in EOS, use `append_eos`
    instead.

    Args:
      dataset: a datasets.Dataset of tokenized examples to preprocess.
      output_features: a mapping of output feature names to Feature objects.
      sequence_length: a mapping from output feature names to max lengths.
        If provided, output feature sequences will be trimmed to ensure they are
        not longer than this length once EOS is added.

    Returns:
      a datasets.Dataset of tokenized examples with EOS added to specified output
      features.
    """

    ret = {}
    for k, v in features.items():
        if k in output_features:
            tokenizer = output_features[k].tokenizer
            eos_token_id = tokenizer.eos_token_id
            if output_features[k].add_eos:
                if eos_token_id is None:
                    raise AttributeError('tokenizer must have eos_token to add eos id but eos token is : {}'.format(
                        eos_token_id))

                if sequence_length is not None:
                    if k in sequence_length:
                        max_length = sequence_length[k]
                        v = v[:max_length-1] + [eos_token_id]
                    else:
                        v = v + [eos_token_id]
                else:
                    v = v + [eos_token_id]
        ret[k] = v
    return ret


@utils.map_over_dataset
def trim_and_pad_output_features(
    features,
    output_features: OutputFeaturesType,
    sequence_length: Optional[SequenceLengthType] = None,
    add_attention_mask: Optional[bool] = True,
) -> datasets.Dataset:
    """Trim and pad first dimension of features to `feature_lengths`.

    Args:
      dataset: tf.data.Dataset, the dataset to trimp/pad examples in.
      feature_lengths: map from feature key to final length. Other features will
        be returned unchanged.
    Returns:
      Trimmed/padded datasets.Dataset.
    """
    ret = {}
    for k, v in features.items():
        if k in output_features:
            tokenizer = output_features[k].tokenizer
            pad_token_id = tokenizer.pad_token_id
            if k in sequence_length:
                length_k = sequence_length[k]
                v = v[:length_k]
                pad_len = length_k - len(v)
                v = v + [pad_token_id] * pad_len
                if add_attention_mask:
                    ret[f'{k}_attention_mask'] = [1]*(length_k - pad_len) + [0] * pad_len
        ret[k] = v
    return ret

