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

import utils

import register_optimizers

flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')

flags.DEFINE_string("task", 'klue_tc',
                    "name of task.")
flags.DEFINE_string("model_name", 'ke_t5.models.models:T5EncoderForSequenceClassificationMean',
                    "name of task.")
flags.DEFINE_string("pre_trained_model", 'KETI-AIR/ke-t5-small',
                    "name or path of pretrained model.")
flags.DEFINE_string("hf_data_dir", '../Korean-Copora/data',
                    "data directory for huggingface dataset."
                    "it is equivalent to the manual directory in tfds."
                    "if you use NIKL dataset, you have to set this variable correctly"
                    "because the data of NIKL has to be downloaded manually.")
flags.DEFINE_string("hf_cache_dir", './cache_dir/huggingface_datasets',
                    "cache directory for huggingface dataset."
                    "it is equivalent to the data directory in tfds.")
flags.DEFINE_string("output_dir", 'output',
                    "path to output directory.")

flags.DEFINE_bool("test", False,
                    "is test mode?.")

flags.DEFINE_string("resume", None,
                    "path to checkpoint.")
flags.DEFINE_string("hf_path", None,
                    "path to score huggingface model")

flags.DEFINE_string("train_split", 'train[:90%]',
                    "name of train split")
flags.DEFINE_string("valid_split", 'train[90%:]',
                    "name of validation split")
flags.DEFINE_integer("batch_size", 16, "mini batch size")
flags.DEFINE_integer("workers", 0, "number of workers for dataloader")
flags.DEFINE_integer("epochs", 3, "number of epochs for training")
flags.DEFINE_integer("start_epoch", 0, "start epoch")
flags.DEFINE_integer("print_freq", 100, "print frequency")
FLAGS = flags.FLAGS



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
    # parsing and binding gin configs
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    # import tasks after binding gin configs
    from ke_t5.task import task

    # override data_dir and cache dir for huggingface datasets
    seq_pipe.set_hf_data_dir_override(FLAGS.hf_data_dir)
    seq_pipe.set_hf_cache_dir_override(FLAGS.hf_cache_dir)

    # get task
    task = seq_pipe.get_task(FLAGS.task)

    # best_function
    best_fn = task.best_fn
    best_score = best_fn.get_min_score()

    # create directory
    path_info = utils.create_directory_info(FLAGS)

    # create metric meter and logger
    metric_meter = utils.MetricMeter(task)
    metric_meter.add_average_meter("loss")
    summary_logger = utils.TensorboardXLogging(path_info["logs_dir"])

    # get model
    model_class = loader.load_model(FLAGS.model_name)
    model_kwargs = task.additional_task_info
    model = model_class.from_pretrained(FLAGS.pre_trained_model, **model_kwargs)
    model = model.cuda()

    # get optimizer
    optimizer_cls = get_optimizer()
    optimizer = optimizer_cls(model.parameters())

    if FLAGS.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(FLAGS.resume):
                print("=> loading checkpoint '{}'".format(FLAGS.resume))
                checkpoint = torch.load(FLAGS.resume)
                FLAGS.start_epoch = checkpoint['epoch']
                best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(FLAGS.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(FLAGS.resume))
        resume()
    
    if FLAGS.hf_path:
        model.save_pretrained(FLAGS.hf_path)
        print('hf model is saved in {}'.format(FLAGS.hf_path))
        exit()
    
    if FLAGS.test:
      test_dataset = get_dataset(task, split=FLAGS.valid_split)
      test_dataset.set_format('torch', columns=task.columns, device='cuda')
      test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size,
                                 shuffle=False, num_workers=FLAGS.workers)
      metric_meter = validate(test_loader, model, 0, FLAGS, metric_meter)
      score_log = metric_meter.get_score_str("test")
      print('-'*10 + 'test'+'-'*10+'\n'+score_log+'-'*24)
      exit()

    # load dataset
    train_dataset = get_dataset(task, split=FLAGS.train_split)
    test_dataset = get_dataset(task, split=FLAGS.valid_split)

    # set dataset as pytorch dataset
    train_dataset.set_format('torch', columns=task.columns, device='cuda')
    test_dataset.set_format('torch', columns=task.columns, device='cuda')

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                  shuffle=True, num_workers=FLAGS.workers)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size,
                                 shuffle=False, num_workers=FLAGS.workers)

    # run training
    for epoch in range(FLAGS.start_epoch, FLAGS.epochs):
      train(train_loader, model, optimizer, epoch, FLAGS, metric_meter, summary_logger)

      metric_meter = validate(test_loader, model, epoch, FLAGS, metric_meter)
      avg_scores = metric_meter.get_average_scores()

      is_best, best_score = best_fn.is_best(avg_scores, best_score)

      utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer' : optimizer.state_dict(),
            }, is_best,
            path_info["ckpt_path"], 
            path_info["best_model_path"])

      summary_logger(
            avg_scores, 
            epoch, 
            FLAGS.task, 
            "eval")


def validate(eval_loader, model, epoch, args, metric_meter):
    batch_time = utils.AverageMeter()
    metric_meter.reset()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in enumerate(eval_loader):
            outputs = model(
                  **batch
              )

            loss = outputs[0]
            logits = outputs[1]

            # update scores
            predictions = utils.get_ids_from_logits(logits)
            metric_meter.update_scores("loss", loss.cpu().numpy())
            metric_meter.update_metrics(batch['labels'], predictions)

            if step_inbatch % args.print_freq == 0:
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                score_log = metric_meter.get_score_str("eval")

                print('-----Evaluation-----\n Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'.format(
                       epoch, step_inbatch, len(eval_loader),
                       args.batch_size/batch_time.val,
                       args.batch_size/batch_time.avg,
                       batch_time=batch_time) + score_log)
    return metric_meter


def train(train_loader, model, optimizer, epoch, args, metric_meter=None, summary_logger=None):
    # calc batch time
    batch_time = utils.AverageMeter()
    metric_meter.reset()
    steps_per_epoch = len(train_loader)

    # switch to train mode
    model.train()
    end = time.time()

    for step_inbatch, batch in enumerate(train_loader):
        outputs = model(
            **batch
        )

        loss = outputs[0]
        logits = outputs[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
          metric_meter.update_scores("loss", loss.cpu().numpy())

          global_step = epoch*args.batch_size + step_inbatch
          if global_step % args.print_freq == 0:
              batch_time.update((time.time() - end)/args.print_freq)
              end = time.time()

              predictions = utils.get_ids_from_logits(logits)

              metric_meter.update_metrics(batch['labels'], predictions)

              score_log = summary_logger(
                    metric_meter.get_average_scores(), 
                    global_step, 
                    args.task, 
                    "train")
              
              print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Speed {3:.3f} ({4:.3f})\t'.format(
                        epoch, step_inbatch, steps_per_epoch,
                        args.batch_size/batch_time.val,
                        args.batch_size/batch_time.avg,
                        batch_time=batch_time)+score_log)


if __name__ == "__main__":
    app.run(main)


# python train.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_param="get_dataset.sequence_length={'inputs':512, 'targets':512}"
# python train.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="train.gin"
