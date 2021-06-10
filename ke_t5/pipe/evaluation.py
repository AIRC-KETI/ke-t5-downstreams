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

import abc
import base64
import concurrent
import inspect
import itertools
import json
import os
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Dict

from absl import logging
import dataclasses
import numpy as np
from tensorboardX import SummaryWriter

from . import dataset_providers

# TODO: Create evaluation helper 

class BestScoreMeta(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_best(self, avg_dict: Dict[str, float], prev_best: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_min_score(self) -> Dict[str, float]:
        raise NotImplementedError

class BestScore(BestScoreMeta):
    def is_best(self, avg_dict: Dict[str, float], prev_best: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        return True, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {'dummy': 0}

class LessIsTheBest(BestScoreMeta):
    def __init__(self, metric_name='loss'):
        self._metric_name = metric_name
    
    def is_best(self, avg_dict, prev_best):
        current_score = avg_dict[self._metric_name]['score']
        prev_best_score = prev_best[self._metric_name]
        is_best = prev_best_score > current_score
        if is_best:
            prev_best[self._metric_name] = current_score
        return is_best, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {self._metric_name: np.inf}

class GreaterIsTheBest(BestScoreMeta):
    def __init__(self, metric_name='accuracy'):
        self._metric_name = metric_name
    
    def is_best(self, avg_dict, prev_best):
        current_score = avg_dict[self._metric_name]['score']
        prev_best_score = prev_best[self._metric_name]
        is_best = prev_best_score < current_score
        if is_best:
            prev_best[self._metric_name] = current_score
        return is_best, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {self._metric_name: -np.inf}


def get_method(o, name):
    return getattr(o, name)

class Evaluator():
    def __init__(self, 
            task_name,
            split,
            log_dir=None,
            local_rank=0,
            world_size=1):
        self._task_name = task_name
        self._task = dataset_providers.get_task(self._task_name)
        self._split = split
        self._log_dir = log_dir
        self._local_rank = local_rank
        self._world_size = world_size

        self._model_fn = 'forward'
    
    def _get_dataset(self):
        return self._task.get_dataset(split=self._split)

    def set_model_fn_for_evaluate(self, model_fn_name='forward'):
        self._model_fn = model_fn_name
    
    def call_model_method(self, model, *args, **kwargs):
        return get_method(model, self._model_fn)(*args, **kwargs)
    
    def evaluate(self):
        dataset = self._get_dataset()
        if self._world_size > 1:
            dataset = dataset.shard(self._world_size, self._local_rank)
    
# evaluator = seq_pipe.Evaluator(
#         task_name=FLAGS.task, 
#         split=FLAGS.valid_split, 
#         log_dir=path_info['logs_dir'] if 'logs_dir' in path_info else None,
#         local_rank=FLAGS.local_rank if FLAGS.distributed else 0,
#         world_size=FLAGS.world_size if FLAGS.distributed else 1)
