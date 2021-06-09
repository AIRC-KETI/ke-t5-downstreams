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

# TODO: Create evaluation helper 

class BestScoreMeta(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_best(self, avg_dict: Dict[str, float], prev_best: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_min_score(self) -> Dict[str, float]:
        raise NotImplementedError

class BestScore(object):
    def is_best(self, avg_dict: Dict[str, float], prev_best: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        return True, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {'dummy': 0}

class LessIsTheBest(BestScore):
    def __init__(self, metric_name='loss'):
        self._metric_name = metric_name
    
    def is_best(self, avg_dict, prev_best):
        current_score = avg_dict[self._metric_name]
        prev_best_score = prev_best[self._metric_name]
        is_best = prev_best_score > current_score
        if is_best:
            prev_best[self._metric_name] = current_score
        return is_best, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {self._metric_name: np.inf}

class GreaterIsTheBest(BestScore):
    def __init__(self, metric_name='accuracy'):
        self._metric_name = metric_name
    
    def is_best(self, avg_dict, prev_best):
        current_score = avg_dict[self._metric_name]
        prev_best_score = prev_best[self._metric_name]
        is_best = prev_best_score < current_score
        if is_best:
            prev_best[self._metric_name] = current_score
        return is_best, prev_best
    
    def get_min_score(self) -> Dict[str, float]:
        return {self._metric_name: -np.inf}


class Evaluator(object):
    def __init__(self, task) -> None:
        super().__init__()
        pass
