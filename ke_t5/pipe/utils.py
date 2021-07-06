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

"""Utilities for data loading and processing."""

import contextlib
import functools
import os
from typing import Mapping, Optional
import logging

import numpy as np

import datasets

from datasets import prepare_module, import_main_class
from datasets.load import url_or_path_parent

_HF_DATA_DIR_OVERRIDE = None
_HF_CACHE_DIR_OVERRIDE = None


def set_hf_data_dir_override(hf_data_dir):
    global _HF_DATA_DIR_OVERRIDE
    _HF_DATA_DIR_OVERRIDE = hf_data_dir


def set_hf_cache_dir_override(hf_cache_dir):
    global _HF_CACHE_DIR_OVERRIDE
    _HF_CACHE_DIR_OVERRIDE = hf_cache_dir


def get_builder(path, name, data_dir=None, cache_dir=None):
    module_path, hash, resolved_file_path = prepare_module(
        path,
        dataset=True,
        return_resolved_file_path=True,
    )

    builder_cls = import_main_class(module_path, dataset=True)
    builder_instance = builder_cls(
        cache_dir=cache_dir,
        name=name,
        data_dir=data_dir,
        hash=hash,
    )
    return builder_instance


class LazyHFDatasetLoader(object):
    """Wrapper for Huggingface datasets with memoization and additional functionality.

    Lazily loads info from TFDS and provides memoization to avoid expensive hidden
    file operations. Also provides additional utility methods.
    """

    _MEMOIZED_BUILDERS = {}

    def __init__(self, path, name, data_dir=None, cache_dir=None, split_map=None, ignore_verifications=True):
        """LazyTfdsLoader constructor.

        Args:
            path: str, Path to the dataset processing script with the dataset builder. 
            name: str, Defining the name of the dataset configuration.
            data_dir: str (optional), Defining the data_dir of the dataset configuration.
            cache_dir: str (optional), Directory to read/write data. Defaults to “~/datasets”.
            split_map: dict (optional), mapping from canonical splits
                (e.g., 'validation') to HF splits or slices
                (e.g., 'train[':1%']).
        """
        self._path = path
        self._name = name
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._split_map = split_map
        self._ignore_verifications = ignore_verifications

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        if _HF_DATA_DIR_OVERRIDE:
            if self._data_dir:
                logging.warning(
                    "Overriding HF data directory '%s' with '%s' for dataset '%s' - '%s'.",
                    self._data_dir, _HF_DATA_DIR_OVERRIDE, self._path, self.name)
            return _HF_DATA_DIR_OVERRIDE
        return self._data_dir

    @property
    def cache_dir(self):
        if _HF_CACHE_DIR_OVERRIDE:
            if self._cache_dir:
                logging.warning(
                    "Overriding HF cache directory '%s' with '%s' for dataset '%s' - '%s'.",
                    self._cache_dir, _HF_CACHE_DIR_OVERRIDE, self._path, self.name)
            return _HF_CACHE_DIR_OVERRIDE
        return self._cache_dir

    @property
    def builder(self):
        builder_key = (self.path, self.name, self.data_dir, self.cache_dir)
        if builder_key not in LazyHFDatasetLoader._MEMOIZED_BUILDERS:
            self.update_builder_key(builder_key)
        return LazyHFDatasetLoader._MEMOIZED_BUILDERS[builder_key]

    @property
    def info(self):
        return self.builder.info

    def update_builder_key(self, builder_key):
        LazyHFDatasetLoader._MEMOIZED_BUILDERS[builder_key] = get_builder(
            self.path, self.name, data_dir=self.data_dir, cache_dir=self.cache_dir)

    def check_builder_key(self):
        builder_key = (self.path, self.name, self.data_dir, self.cache_dir)
        if builder_key not in LazyHFDatasetLoader._MEMOIZED_BUILDERS:
            self.update_builder_key(builder_key)
        elif LazyHFDatasetLoader._MEMOIZED_BUILDERS[builder_key] is None:
            self.update_builder_key(builder_key)

    def _map_split(self, split):
        return self._split_map[split] if self._split_map else split

    def load(self, split):
        """Returns a huggingface dataset for the given split."""

        loaded_dataset = datasets.load_dataset(
            self._path,
            self._name,
            split=split,
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            ignore_verifications=self._ignore_verifications
        )
        self.check_builder_key()
        return loaded_dataset

    def size(self, split):
        """Returns the number of examples in the split."""
        split = self._map_split(split)
        ds_splits = self.info.splits
        dataset_size = None
        if ds_splits is not None:
            dataset_size = ds_splits[split].num_examples
            dataset_size = dataset_size if dataset_size > 0 else np.inf
        return dataset_size


def map_over_dataset(fn):
    """Decorator to map decorated function over dataset.

    Many preprocessors map a function over a dataset. This decorator helps reduce
    boilerplate for this common pattern.

    Args:
      fn: map function

    Returns:
      Function which takes dataset as first argument.
    """

    @functools.wraps(fn)
    def wrapped_fn(ds, *args, num_proc=1, **kargs):
        return ds.map(
            lambda arg: fn(arg, *args, **kargs), num_proc=num_proc)

    return wrapped_fn


def map_over_dataset_batched(fn):
    """Decorator to map decorated function over dataset.

    Many preprocessors map a function over a dataset. This decorator helps reduce
    boilerplate for this common pattern.

    Args:
      fn: map function

    Returns:
      Function which takes dataset as first argument.
    """

    @functools.wraps(fn)
    def wrapped_fn(ds, *args, num_proc=1, **kargs):
        return ds.map(
            lambda arg: fn(arg, *args, **kargs), num_proc=num_proc, batched=True)

    return wrapped_fn

def map_over_dataset_batched_format(fn=None, *, batch_size=100, remove_columns=None):
    """Decorator to map decorated function over dataset.

    Many preprocessors map a function over a dataset. This decorator helps reduce
    boilerplate for this common pattern.

    Args:
      fn: map function

    Returns:
      Function which takes dataset as first argument.
    """

    def map_with_option(fn):
        @functools.wraps(fn)
        def wrapped_fn(ds, *args, num_proc=1, **kargs):
            print(remove_columns)
            return ds.map(
                lambda arg: fn(arg, *args, **kargs), num_proc=num_proc, batched=True, batch_size=batch_size, remove_columns=remove_columns)
        return wrapped_fn

    if fn is None:
        return map_with_option
    else:
        return map_with_option(fn)


def filter_over_dataset(fn):

    @functools.wraps(fn)
    def wrapped_fn(ds, *args, num_proc=1, **kargs):
        return ds.map(
            lambda arg: fn(arg, *args, **kargs), num_proc=num_proc)

    return wrapped_fn

# if __name__ == "__main__":
#     path = 'KETI-AIR/klue'
#     #name='ne.2020.v1.0'
#     name='tc'
#     data_dir="../Korean-Copora/data"
#     #cache_dir="../Korean-Corpora/cache_dir/huggingface_datasets"
#     cache_dir="./cache_dir/huggingface_datasets"


#     ds = LazyHFDatasetLoader(path, name, data_dir=data_dir, cache_dir=cache_dir)
#     dataset = ds.load('train')

#     print(dataset[0])

#     print(ds.size('train[:10%]'))
