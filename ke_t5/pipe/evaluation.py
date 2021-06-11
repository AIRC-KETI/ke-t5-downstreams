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

import gin
from absl import logging
import dataclasses
from gin.config import configurable
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

@gin.configurable
class EvaluationHelper(object):
    def __init__(self, 
            task,
            model_fn=None,
            model_input_keys=None,
            model_kwargs=None,
            distributed=True):

        self._task = task

        self._model_fn = model_fn if model_fn is not None else 'forward'
        self._model_input_keys = model_input_keys if model_input_keys is not None else ['input_ids']
        self._model_kwargs = model_kwargs

        if self._model_kwargs is None:
            if 'task_specific_params' in self._task.additional_task_info:
                self._model_kwargs = list(self._task.additional_task_info['task_specific_params'].values())[0]
                logging.info(f'model kwargs: {self._model_kwargs}')
            else:
                {}

        self._logit_to_id = task.logit_to_id
        self._distributed = distributed

        self._model_fn_params = None

    @property
    def logit_to_id(self):
        if self._model_fn == 'forward':
            return self._logit_to_id
        elif self._model_fn == 'generate':
            return False
        return False

    def select_model_inputs(self, data):
        return {k:v for k, v in data.items() if k in self._model_input_keys}
    
    def call_model_method(self, model, *args, **kwargs):
        if self._distributed:
            model = model.module

        if self._model_fn_params is None:
            model_fn = get_method(model, self._model_fn)
            self._model_fn_params = list(inspect.signature(model_fn).parameters.keys())
            logging.info(f'set model params {self._model_fn_params}')

        # filter input keys
        kwargs = {k:v for k, v in kwargs.items() if k in self._model_fn_params}

        return get_method(model, self._model_fn)(*args, **kwargs)
    
    def prepare_inputs(self, data):
        kw_inputs = self.select_model_inputs(data)
        if self._model_kwargs is not None:
            for k, v in self._model_kwargs.items():
                kw_inputs[k] = v
        return kw_inputs
    
    
# evaluator = seq_pipe.Evaluator(
#         task_name=FLAGS.task, 
#         split=FLAGS.valid_split, 
#         log_dir=path_info['logs_dir'] if 'logs_dir' in path_info else None,
#         local_rank=FLAGS.local_rank if FLAGS.distributed else 0,
#         world_size=FLAGS.world_size if FLAGS.distributed else 1)
