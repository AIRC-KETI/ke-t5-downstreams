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

import os
import abc
import shutil
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import dataclasses

from tensorboardX import SummaryWriter

import torch


# adopted from 'seqio' github


@dataclasses.dataclass
class Metric:
    """A base method for the dataclasses that represent tensorboard values.

    Task `metric_fn`s should output `Mapping[str, Metric]` which will be written
    to tensorboard. `Metric` subclasses are used to dispatch to the correct
    tensorboard writing function.
    """


# adopted from 'seqio' github
@dataclasses.dataclass
class Scalar(Metric):
    """The default tensorflow value, used for creating time series graphs."""
    value: float


def make_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def create_directory_info(args, create_dir=True):

    model_dir = os.path.join(args.output_dir, "{}_{}".format(
        args.model_name.replace('/', '_'), args.pre_trained_model.replace('/', '_')), args.task)
    weights_dir = os.path.join(model_dir, "weights")
    logs_dir = os.path.join(model_dir, "logs")

    path_info = {
        'weights_dir': weights_dir,
        'logs_dir': logs_dir,
    }

    if create_dir:
        for k, v in path_info.items():
            make_dir_if_not_exist(v)

    path_info['best_model_path'] = os.path.join(weights_dir, "best_model.pth")
    path_info['ckpt_path'] = os.path.join(weights_dir, "checkpoint.pth")
    return path_info


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def get_ids_from_logits(logits):
    _, predicted = torch.max(logits, -1)
    return predicted


class MetricMeter(object):
    def __init__(self, task) -> None:
        super().__init__()
        self.train_postprocess_fn = task.train_postprocess_fn
        self.predict_metric_fns = task.metric_fns

        self._average_meters = {}

    def _get_averagemeter(self, metric_name):
        self.add_average_meter(metric_name)
        return self._average_meters[metric_name]

    def add_average_meter(self, metric_name):
        if metric_name not in self._average_meters:
            self._average_meters[metric_name] = AverageMeter()

    def update_metrics(self, gathered_dict):
        if self.train_postprocess_fn is not None:
            gathered_dict = self.train_postprocess_fn(gathered_dict)

        for predict_metric_fn in self.predict_metric_fns:
            scores_dict = predict_metric_fn(gathered_dict)
            for score_name, score in scores_dict.items():
                average_meter = self._get_averagemeter(score_name)
                average_meter.update(score['score'], score['count'])

    def update_scores(self, score_name, score):
        average_meter = self._get_averagemeter(score_name)
        average_meter.update(score['score'], score['count'])

    def reset(self, name=None):
        if name is not None:
            if name in self._average_meters:
                self._average_meters[name].reset()
        else:
            for am_name in self._average_meters.keys():
                self._average_meters[am_name].reset()
    
    def set_average_scores(self, average_scores):
        for k, v in average_scores.items():
            average_meter = self._get_averagemeter(k)
            average_meter.update(v['score'], v['count'])

    def get_average_scores(self):
        return {k: {'score':v.avg, 'count': v.count} for k, v in self._average_meters.items()}

    def get_score_str(self, tag_name='eval', average_scores=None):
        if average_scores is None:
            return ''.join([
                "{tag_name}/{tag}: {metric_value:.3f}\t".format(tag_name=tag_name, tag=k, metric_value=v['score']) for k, v in self.get_average_scores().items()])
        else:
            return ''.join([
                "{tag_name}/{tag}: {metric_value:.3f}\t".format(tag_name=tag_name, tag=k, metric_value=v['score']) for k, v in average_scores.items()])


class Logger(abc.ABC):

    @abc.abstractmethod
    def __call__(self,
                 task_metrics: Mapping[str, float],
                 step: int,
                 task_name: str) -> None:
        """Logs the metric for each task."""

    @abc.abstractproperty
    def summary_dir(self) -> str:
        pass


class TensorboardXLogging(Logger):
    def __init__(self, summary_dir: str):
        self._summary_dir = summary_dir
        self._summary_writers = {}

    @property
    def summary_dir(self) -> str:
        return self._summary_dir

    def _get_summary_writer(self, task_name: str):
        if task_name not in self._summary_writers:
            self._summary_writers[task_name] = SummaryWriter(
                os.path.join(self._summary_dir, task_name)
            )
        return self._summary_writers[task_name]

    def __call__(self, task_metrics: Mapping[str, Mapping[str, Scalar]], step: int,
                 task_name: str, tag_name='eval') -> str:
        summary_writer = self._get_summary_writer(task_name)

        metric_str = ''

        for metric_name, metric_value in task_metrics.items():
            tag = f"{tag_name}/{metric_name}"

            summary_writer.add_scalar(tag,
                                      torch.tensor(metric_value['score']), step)
            metric_str += f"{tag}: {metric_value['score']:.3f}\t"
        return metric_str


import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


collate_variable_length_err_msg_format = (
    "collate_variable_length: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_variable_length_dict_outer(batch):
    elem = batch[0]

    if isinstance(elem, collections.abc.Mapping):
        return {key: _collate_variable_length_dict([d[key] for d in batch]) for key in elem}
    else:
        raise RuntimeError('each element must be the Dictionary types')

# claate variable length for dictionary
def _collate_variable_length_dict(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_variable_length_err_msg_format.format(elem.dtype))

            return _collate_variable_length_dict([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: [d[key] for d in batch] for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_collate_variable_length_dict(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [_collate_variable_length_dict(samples) for samples in transposed]

    raise TypeError(collate_variable_length_err_msg_format.format(elem_type))


def collate_variable_length(batch):
    elem = batch[0]

    if isinstance(elem, collections.abc.Mapping):
        return {key: _collate_variable_length([d[key] for d in batch]) for key in elem}
    else:
        raise RuntimeError('each element must be the Dictionary types')


def _collate_variable_length(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if len(batch) == 0:
        return []

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_variable_length_err_msg_format.format(elem.dtype))

            return _collate_variable_length([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: [d[key] for d in batch] for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_collate_variable_length(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        return [_collate_variable_length(elem) for elem in it]

    raise TypeError(collate_variable_length_err_msg_format.format(elem_type))