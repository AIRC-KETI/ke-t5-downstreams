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
from absl import logging

import torch
from torch import is_tensor, tensor
from torch.utils.data import DataLoader

# distributed
import torch.utils.data

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
flags.DEFINE_string("hf_data_dir", './data',
                    "data directory for huggingface dataset."
                    "it is equivalent to the manual directory in tfds."
                    "if you use NIKL dataset, you have to set this variable correctly"
                    "because the data of NIKL has to be downloaded manually.")
flags.DEFINE_string("hf_cache_dir", './cache_dir/huggingface_datasets',
                    "cache directory for huggingface dataset."
                    "it is equivalent to the data directory in tfds.")
flags.DEFINE_string("output_dir", 'output',
                    "path to output directory.")

flags.DEFINE_string("resume", None,
                    "path to checkpoint.")
flags.DEFINE_string("hf_path", None,
                    "path to score huggingface model")


flags.DEFINE_string("test_split", 'test',
                    "name of test split")
flags.DEFINE_integer("batch_size", 16, "mini batch size")
flags.DEFINE_integer("workers", 0, "number of workers for dataloader")
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
    # pretrained model will be downloaded if there is no cache
    from ke_t5.task import task

    # override data_dir and cache dir for huggingface datasets
    seq_pipe.set_hf_data_dir_override(FLAGS.hf_data_dir)
    seq_pipe.set_hf_cache_dir_override(FLAGS.hf_cache_dir)

    # get task
    task = seq_pipe.get_task(FLAGS.task)

    metric_meter = utils.MetricMeter(task)
    path_info = utils.create_directory_info(FLAGS, create_dir=False)

    # get model
    model_class = loader.load_model(FLAGS.model_name)
    model_kwargs = task.additional_task_info
    if FLAGS.hf_path:
        model = model_class.from_pretrained(FLAGS.hf_path)
        logging.info('load huggingface model from {}'.format(FLAGS.hf_path))
    else:
        model = model_class.from_pretrained(
            FLAGS.pre_trained_model, **model_kwargs)
        logging.info('create model from huggingface {}'.format(FLAGS.pre_trained_model))
    
    model = model.cuda()

    if FLAGS.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(FLAGS.resume):
                logging.info("=> loading checkpoint '{}'".format(FLAGS.resume))
                checkpoint = torch.load(
                    FLAGS.resume, map_location="cuda")
                model.load_state_dict(checkpoint['state_dict'])
                logging.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(FLAGS.resume, checkpoint['epoch']))
            elif FLAGS.resume.lower()=='true':
                FLAGS.resume = path_info['best_model_path']
                resume()
            else:
                logging.info("=> no checkpoint found at '{}'".format(FLAGS.resume))
        resume()
    
    eval_helper = seq_pipe.EvaluationHelper(
      task=task,
      distributed=False
    )

    
    test_dataset = get_dataset(task, split=FLAGS.test_split)
    test_dataset.set_format('torch', columns=task.model_input_columns, device='cuda', output_all_columns=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=FLAGS.batch_size, 
        shuffle=False, 
        num_workers=FLAGS.workers)
    metric_meter = validate(test_loader, model, eval_helper, FLAGS, metric_meter)

    score_log = metric_meter.get_score_str("test")
    logging.info('\n' + '-'*10 + 'test'+'-'*10+'\n'+score_log+ '\n' + '-'*24)


def validate(eval_loader, model, eval_helper, args, metric_meter):
    batch_time = utils.AverageMeter()
    
    # reset metric_meter
    metric_meter.reset()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in enumerate(eval_loader):
            # select model inputs
            inputs = eval_helper.prepare_inputs(batch)

            # call model_fns
            outputs = eval_helper.call_model_method(model, **inputs)


            # get predictions
            if eval_helper.logit_to_id:
                logits = outputs[0]
                predictions = utils.get_ids_from_logits(logits.clone())
            else:
                predictions = outputs.clone()

            # update metrics
            predictions = predictions.cpu().numpy()
            gathered_dict = {k:v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
            gathered_dict['predictions'] = predictions
            metric_meter.update_metrics(gathered_dict)

            # reduce average scores
            average_scores = metric_meter.get_average_scores()
            if step_inbatch % args.print_freq == 0:
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                if args.local_rank == 0 or not args.distributed:
                    score_log = metric_meter.get_score_str("eval", average_scores=average_scores)

                    logging.info('-----Evaluation----- \nSteps: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {2:.3f} ({3:.3f})\t'.format(
                                step_inbatch, len(eval_loader),
                                args.batch_size/batch_time.val,
                                args.batch_size/batch_time.avg,
                                batch_time=batch_time) + score_log)

    return metric_meter


if __name__ == "__main__":
    app.run(main)


# python test.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="test.gin" --model_name="transformers:AutoModelForSequenceClassification" --pre_trained_model="bert-base-uncased"
# python test.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_param="get_dataset.sequence_length={'inputs':512, 'targets':512}" --model_name="transformers:AutoModelForSequenceClassification" --pre_trained_model="bert-base-uncased"

# python test.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="test.gin"
# python test.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="test.gin" --resume output/ke_t5.models.models:T5EncoderForSequenceClassificationMean_KETI-AIR/ke-t5-small/weights/best_model.pth --test true --test_split "test"


# python test.py --gin_file="test.gin" --task 'klue_tc'
# python test.py --gin_file="test.gin" --task 'klue_re'
# python test.py --gin_file="test.gin" --task 'klue_nli'

# python test.py --gin_file="test.gin" --model_name transformers:T5ForConditionalGeneration --task 'klue_nli_gen'
# python test.py --gin_file="test.gin" --model_name transformers:T5ForConditionalGeneration --task 'nikl_summarization_topic'


# python test.py --gin_file="test.gin" --model_name transformers:T5ForConditionalGeneration --task 'korquad_gen'
# python test.py --gin_file="test.gin" --model_name ke_t5.models.models:T5EncoderForSequenceClassificationMean --task 'klue_tc'

# python test.py --gin_file="test.gin" --model_name transformers:T5ForConditionalGeneration --task 'klue_tc_gen' --resume true


# training
# python train.py --gin_file="train.gin" --model_name ke_t5.models.models:T5EncoderForSequenceClassificationMean --task 'klue_tc' --train_split train --valid_split test --epochs 5

# test
# python test.py --gin_file="test.gin" --model_name ke_t5.models.models:T5EncoderForSequenceClassificationMean --task 'klue_tc' --test_split test --resume true

