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

# This code is modified from 'seqio' for huggingface datasets

import abc
import collections
import functools
import inspect
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

import dataclasses

import torch
import datasets
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from ke_t5.pipe import utils

_VALID_TASK_NAME_REGEX = re.compile(r"^[\w\d\.\:_]+$")


@dataclasses.dataclass(frozen=True)
class Feature:
    """A container for attributes of output features that have to tokenize"""
    tokenizer: PreTrainedTokenizerBase
    add_eos: bool = True
    required: bool = True
    dtype: str = 'torch'


class DatasetProviderBase(metaclass=abc.ABCMeta):
    """Abstract base for classes that provide a tf.data.Dataset."""

    @abc.abstractproperty
    def output_features(self) -> Mapping[str, Feature]:
        raise NotImplementedError

    @abc.abstractproperty
    def splits(self) -> Sequence[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(
        self,
        sequence_length: int,
        split: str,
        use_cached: bool = True,
    ) -> Any:
        """Returns the requested tf.data.Dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def num_input_examples(self, split: str) -> int:
        raise NotImplementedError


class DatasetProviderRegistry(object):
    """Base for registry of data providers.

    Subclasses must wrap `get` method to override the return type for pytype.
    TODO(adarob): Remove the need to override `get`.
    """
    # Class variables must be defined in subclasses.
    _REGISTRY: MutableMapping[str, DatasetProviderBase]
    _PROVIDER_TYPE: Type[DatasetProviderBase]

    @classmethod
    def add_provider(cls, name: str, provider):
        """Adds a data provider instance to the registry."""
        if name in cls._REGISTRY:
            raise ValueError(
                "Attempting to register duplicate provider: %s" % name)
        if not isinstance(provider, cls._PROVIDER_TYPE):
            raise ValueError(
                "Attempting to register a class not of an invalid type. "
                "Expecting instance of %s, got %s" %
                (cls._PROVIDER_TYPE, type(provider).__name__))

        cls._REGISTRY[name] = provider

    @classmethod
    def add(
        cls,
        name: str,
        provider_cls,
        *provider_args,
        **provider_kwargs
    ):
        """Instantiates and adds provider to the registry."""
        if not issubclass(provider_cls, cls._PROVIDER_TYPE):
            raise ValueError(
                "Attempting to register a class not of an invalid type. "
                "Expecting instance of %s, got %s" %
                (cls._PROVIDER_TYPE, provider_cls))
        provider = provider_cls(*provider_args, **provider_kwargs)
        cls.add_provider(name, provider)
        return provider

    @classmethod
    def remove(cls, name):
        """Remove provider from the registry, if it exists."""
        if name in cls._REGISTRY:
            del cls._REGISTRY[name]

    @classmethod
    def get(cls, name):
        """Returns provider from the registry."""
        if name not in cls._REGISTRY:
            raise ValueError("Provider name not registered: %s" % name)
        return cls._REGISTRY[name]

    @classmethod
    def names(cls):
        """Returns all provider names in registry."""
        return cls._REGISTRY.keys()

    @classmethod
    def reset(cls):
        """Removes all of the registered tasks."""
        cls._REGISTRY = {}

    @classmethod
    def get_dataset(
            cls,
            name,
            sequence_length,
            split,
            use_cached=True):
        """Returns the requested tf.data.Dataset."""
        return cls.get(name).get_dataset(
            sequence_length=sequence_length, split=split, use_cached=use_cached)


class DataSource(DatasetProviderBase):
    """A `DatasetProvider` that provides raw data from an input source.

    Inherits all abstract methods and properties of `DatasetProviderBase` except
    those overidden below.
    """

    def __init__(
            self,
            splits: Iterable[str],
            num_input_examples: Optional[Mapping[str, int]] = None):
        self._splits = tuple(splits)
        self._num_input_examples = (
            dict(num_input_examples) if num_input_examples is not None else None)

    @property
    def splits(self) -> Sequence[str]:
        return self._splits

    @property
    def output_features(self) -> Mapping[str, Feature]:
        """Override unused property of `DatasetProviderBase`."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(
        self,
        split: str,
    ) -> Any:
        """
        Args:
          split: string, the split to return.
        """
        raise NotImplementedError

    def num_input_examples(self, split: str) -> Optional[int]:
        if self._num_input_examples is None:
            return None
        return self._num_input_examples[split]


class HFDataSource(DataSource):
    """A `DataSource` that uses TensorFlow Datasets to provide the input data."""

    def __init__(
        self,
        hf_path: str,
        hf_name: str,
        hf_data_dir: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        splits: Optional[Union[Iterable[str], Mapping[str, str]]] = None,
        ignore_verifications: bool = True
    ):
        """HFdataset constructor.

        Args:
            hf_path: string, the name and version number of a HF dataset,
                optionally with a config.
            hf_name: string, the name and version number of a HF dataset,
                optionally with a config.
            hf_data_dir: string, an optional path to a specific HF data directory
                to use.
            hf_cache_dir: string, an optional path to a specific cache directory 
                that is required to create dataset
            splits: an iterable of allowable string split names, a dict mapping
                allowable canonical splits (e.g., 'validation') to HF splits or slices
                (e.g., 'train[':1%']), or None. The default, None, uses all available
                splits from the HF dataset info.
        """

        if splits and not isinstance(splits, dict):
            splits = {k: k for k in splits}

        self._hf_dataset = utils.LazyHFDatasetLoader(
            hf_path,
            hf_name,
            data_dir=hf_data_dir,
            cache_dir=hf_cache_dir,
            split_map=splits if isinstance(splits, dict) else None,
            ignore_verifications=ignore_verifications)

        # If splits are not provided, we pass an empty tuple and use the lazy
        # lookup in the `splits` property.
        super().__init__(splits=splits or ())

    @property
    def splits(self):
        return self._splits or self._hf_dataset.info.splits

    @property
    def hf_dataset(self):
        return self._hf_dataset

    def get_dataset(
        self,
        split: str,
    ) -> Any:
        return self._hf_dataset.load(split)

    def num_input_examples(self, split: str) -> int:
        """Overrides since we can't call `info.splits` until after init."""
        return self.hf_dataset.size(split)


MetricFnCallable = Callable[..., Mapping[str, float]]


class Task(DatasetProviderBase):
    """A class to manage a dataset and its related metrics."""

    def __init__(
            self,
            name: str,
            source: DataSource,
            output_features: Mapping[str, Feature],
            preprocessors: Optional[Sequence[Callable[..., Dataset]]] = None,
            postprocess_fn: Optional[Callable[..., Any]] = None,
            metric_fns: Optional[Sequence[MetricFnCallable]] = None,
            num_proc: Optional[int] = None):
        """Task constructor.

        Args:
          name: a unique name for the Task.
          source: a `DataSource` that provides a raw `datasets.Dataset`.
          output_features: dict(str, Feature), output features of the Task to be
            passed to the model. After preprocessing, examples will be validated to
            ensure they include features that match this specification. Note that
            additional features may be included (e.g., for evaluation), but they
            will not be passed to the model.
          preprocessors: list(callable), an optional list of functions that receive
            a datasets.Dataset and return a datasets.Dataset. These will be executed
            sequentually and the final dataset must include features matching
            `output_features`.
          postprocess_fn: callable, an optional function that receives decoded model
            outputs and converts them to a form that is ready for evaluation using
            the metric functions in `metric_fns`.
          metric_fns: list(callable), an optional list of metric functions with the
            signature `metric_fn(targets, predictions)` to use during evaluation. If
            undefined or empty, no evaluation will occur on the task.
        """
        if not _VALID_TASK_NAME_REGEX.match(name):
            raise ValueError(
                "Task name '%s' contains invalid characters. Must match regex: %s" % (
                    name, _VALID_TASK_NAME_REGEX.pattern))

        metric_fns = metric_fns or []
        self._predict_metric_fns = []
        self._score_metric_fns = []
        for metric_fn in metric_fns:
            pos_args = tuple(
                key for key, param in inspect.signature(metric_fn).parameters.items()
                if param.default == inspect.Parameter.empty
            )
            if pos_args == ("targets", "scores"):
                self._score_metric_fns.append(metric_fn)
            elif pos_args == ("targets", "predictions"):
                self._predict_metric_fns.append(metric_fn)
            else:
                raise ValueError(
                    "Metric functions must have positional arguments matching either "
                    "('targets', 'predictions') or ('targets', 'scores'). "
                    f"Got: {pos_args}")

        self._name = name
        self._source = source

        self._num_proc = num_proc

        preprocessors = tuple(preprocessors or [])

        self._preprocessors = preprocessors

        self._metric_fns = tuple(metric_fns)
        self._postprocess_fn = postprocess_fn

        self._stats = {}

        self._output_features = collections.OrderedDict(
            sorted(list(output_features.items()))
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def metric_fns(self) -> Sequence[MetricFnCallable]:
        """List of all metric functions."""
        return self._predict_metric_fns + self._score_metric_fns

    @property
    def score_metric_fns(self) -> Sequence[MetricFnCallable]:
        """List of metric functions that use log likelihood scores."""
        return self._score_metric_fns

    @property
    def predict_metric_fns(self) -> Sequence[MetricFnCallable]:
        """List of metric functions that use model predictions."""
        return self._predict_metric_fns

    @property
    def output_features(self) -> Mapping[str, Feature]:
        return self._output_features

    @property
    def splits(self) -> Sequence[str]:
        s = self.source.splits
        if not s:
            raise ValueError(f"Task {self.name} has no splits")
        return s

    @property
    def source(self) -> DataSource:
        return self._source

    @property
    def preprocessors(self) -> Sequence[Callable[..., Dataset]]:
        return self._preprocessors

    def num_input_examples(self, split: str) -> Optional[int]:
        return self.source.num_input_examples(split)

    def _preprocess_dataset(
            self,
            dataset: Any,
            preprocessors: Sequence[Callable[..., Dataset]],
            sequence_length: Optional[Mapping[str, int]] = None) -> Any:
        """Sequentially applies preprocessors."""
        for prep_fn in preprocessors:
            # prep_fn must not rely on variable length keyword args such as **kwargs.
            fn_args = set(inspect.signature(prep_fn).parameters.keys())
            kwargs = {}
            if "sequence_length" in fn_args:
                kwargs["sequence_length"] = sequence_length
            if "output_features" in fn_args:
                kwargs["output_features"] = self.output_features
            if "num_proc" in fn_args:
                kwargs["num_proc"] = self._num_proc
            dataset = prep_fn(dataset, **kwargs)
        return dataset

    def get_dataset(
        self,
        sequence_length: Optional[Mapping[str, int]],
        split: str = datasets.Split.TRAIN,
    ) -> Dataset:

        source = self.source

        ds = source.get_dataset(split=split)

        ds = self._preprocess_dataset(
            ds,
            self._preprocessors,
            sequence_length=sequence_length,
        )

        return ds

    def postprocess_fn(self, decoded_model_output: Any,
                       **postprocess_kwargs) -> Any:
        """Returns the model output after applying the postprocess function."""
        if self._postprocess_fn:
            return self._postprocess_fn(decoded_model_output, **postprocess_kwargs)
        return decoded_model_output


class TaskRegistry(DatasetProviderRegistry):
    """Registry of Tasks."""
    _REGISTRY = {}
    _PROVIDER_TYPE = Task

    @classmethod
    def add(
            cls,
            name: str,
            source: DataSource,
            output_features: Mapping[str, Feature],
            preprocessors: Optional[Sequence[Callable[..., datasets.Dataset]]] = None,
            postprocess_fn: Optional[Callable[..., Any]] = None,
            metric_fns: Optional[Sequence[Callable[..., Mapping[str, float]]]] = None,
            **kwargs) -> Task:
        """See `Task` constructor for docstring."""
        return super().add(name, Task, name, source, output_features, preprocessors,
                           postprocess_fn, metric_fns, **kwargs)

    @classmethod
    def get(cls, name) -> Task:
        return super().get(name)


def get_task(task_name):
    """Return the Task from the appropriate registry."""
    tasks = TaskRegistry.names()
    if task_name in tasks:
        return TaskRegistry.get(task_name)
    else:
        raise ValueError("No Task found with name: %s" %
                         task_name)


if __name__ == "__main__":
    path = 'KETI-AIR/klue'
    # name='ne.2020.v1.0'
    name = 'ner'
    data_dir = "../Korean-Copora/data"
    # cache_dir="../Korean-Corpora/cache_dir/huggingface_datasets"
    cache_dir = "./cache_dir/huggingface_datasets"

    # ds = HFDataSource(
    #     path, name, hf_data_dir=data_dir, hf_cache_dir=cache_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('KETI-AIR/ke-t5-small')

    from typing import List

    import types

    # def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
    #     return token_ids

    # if hasattr(tokenizer, "_add_eos_if_not_present") and callable(tokenizer._add_eos_if_not_present):
    #     #tokenizer._add_eos_if_not_present = types.MethodType(_add_eos_if_not_present, tokenizer)
    #     tokenizer._add_eos_if_not_present = functools.partial(_add_eos_if_not_present, self=tokenizer)

    DEFAULT_OUTPUT_FEATURES = {
        "text": Feature(
            tokenizer=tokenizer, add_eos=False, required=False, dtype='torch')
    }

    from ke_t5.pipe.utils import map_over_dataset

    @map_over_dataset
    def add_guid_txt(example):
        example['guid'] = example['guid'] + '-hey'
        return example

    @map_over_dataset
    def add_prefix(example, output_features=None):
        example['guid'] = 'hey-' + example['guid']
        return example

    from ke_t5.pipe.preprocessors import tokenize_output_features, tokenize_and_append_eos_output_features

    task = Task(
        "klue_tc",
        HFDataSource(path, name, hf_data_dir=data_dir,
                     hf_cache_dir=cache_dir),
        DEFAULT_OUTPUT_FEATURES,
        preprocessors=[
            add_prefix,
            add_guid_txt,
            tokenize_and_append_eos_output_features,
        ],
        num_proc=1
    )
    ds = task.get_dataset('train')

    print(ds[0])
    print(ds._fingerprint)
    print(ds.cache_files)
