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

from __future__ import print_function, division
import os
from re import split
import time
import shutil
import argparse

import gin
import gin.torch

from absl import app
from absl import flags

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from ke_t5.task.utils import get_vocabulary
from ke_t5 import pipe as seq_pipe
from ke_t5.models import loader

flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')

flags.DEFINE_string("task", 'klue_tc',
                    "name of task.")
flags.DEFINE_string("model_name", 'ke_t5.models.models:T5EncoderForSequenceClassificationMean',
                    "name of task.")
flags.DEFINE_string("pre_trained_model", 'KETI-AIR/ke-t5-base',
                    "name or path of pretrained model.")
flags.DEFINE_string("hf_data_dir", '../Korean-Copora/data',
                    "data directory for huggingface dataset."
                    "it is equivalent to the manual directory in tfds."
                    "if you use NIKL dataset, you have to set this variable correctly"
                    "because the data of NIKL has to be downloaded manually.")
flags.DEFINE_string("hf_cache_dir", './cache_dir/huggingface_datasets',
                    "cache directory for huggingface dataset."
                    "it is equivalent to the data directory in tfds.")

FLAGS = flags.FLAGS

gin.external_configurable(torch.optim.AdamW)
gin.external_configurable(torch.optim.SparseAdam)

@gin.configurable
def get_dataset(task, sequence_length=None, split=None):
    return task.get_dataset(
            sequence_length=sequence_length,
            split=split
        )

@gin.configurable
def get_optimizer(optimizer_cls):
    return optimizer_cls

def main(_):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    from ke_t5.task import task

    seq_pipe.set_hf_data_dir_override(FLAGS.hf_data_dir)
    seq_pipe.set_hf_cache_dir_override(FLAGS.hf_cache_dir)

    task = seq_pipe.get_task(FLAGS.task)
    
    train_dataset = get_dataset(task, split='train')
    model_class = loader.load_model(FLAGS.model_name)

    model_kwargs = task.additional_task_info
    model = model_class.from_pretrained(FLAGS.pre_trained_model, **model_kwargs)
    model = model.cuda()

    optimizer_cls = get_optimizer()

    #print(model.config.num_labels)


if __name__ == "__main__":
    app.run(main)


# python train.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_param="get_dataset.sequence_length={'inputs':512, 'targets':512}"
# python train.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="train.gin"