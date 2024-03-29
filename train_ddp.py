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
import time
import importlib


import gin
import gin.torch

from absl import app
from absl import flags
from absl import logging

import torch
from torch.utils.data import DataLoader
from torch import optim

# distributed
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

from ke_t5.task.utils import get_vocabulary
from ke_t5 import pipe as seq_pipe
from ke_t5.models import loader, models


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
flags.DEFINE_string("pre_trained_model", 'KETI-AIR/ke-t5-base',
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

flags.DEFINE_bool("test", False,
                  "is test mode?.")
flags.DEFINE_bool("pass_only_model_io", False,
                  "filter all the feature keys except model io features.")
flags.DEFINE_bool("schedule", False,
                  "learning rate schedule")

flags.DEFINE_string("resume", None,
                    "path to checkpoint.")
flags.DEFINE_string("hf_path", None,
                    "path to score huggingface model")

flags.DEFINE_string("train_split", 'train[:90%]',
                    "name of train split")
flags.DEFINE_string("valid_split", 'train[90%:]',
                    "name of validation split")
flags.DEFINE_integer("batch_size", 32, "mini batch size")
flags.DEFINE_integer("workers", 0, "number of workers for dataloader")
flags.DEFINE_integer("epochs", 30, "number of epochs for training")
flags.DEFINE_integer("start_epoch", 0, "start epoch")
flags.DEFINE_integer("print_freq", 50, "print frequency")
flags.DEFINE_float("learning_rate", 0.001, "max learning rate for linear anealing scheduler")


flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_integer("gpu", 0, "gpu id to run")
flags.DEFINE_integer(
    "world_size", 1, "world size. (num_nodes*num_dev_per_node)")
flags.DEFINE_boolean("distributed", True, "is distributed training")
flags.DEFINE_integer("local_rank", 0, "local rank for disributed training.")

FLAGS = flags.FLAGS


@gin.configurable
def get_dataset(task, sequence_length=None, split=None):
    return task.get_dataset(
        sequence_length=sequence_length,
        split=split
    )

@gin.configurable
def get_optimizer(optimizer_cls=torch.optim.AdamW):
    return optimizer_cls


def main(_):
    # check world size
    FLAGS.distributed = False
    if 'WORLD_SIZE' in os.environ:
        FLAGS.distributed = int(os.environ['WORLD_SIZE']) > 1

    # run it on the main proc in distributed settings
    if FLAGS.local_rank == 0 and FLAGS.distributed:
        # parsing and binding gin configs
        gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

        # import tasks after binding gin configs
        # tokenizer for pretrained model will be downloaded if there is no cache
        # downloading tokenizer for pretrained model is only required once,
        # this code will be excuted on the main proc in the distributed setting
        from ke_t5.task import task

        # override data_dir and cache dir for huggingface datasets
        seq_pipe.set_hf_data_dir_override(FLAGS.hf_data_dir)
        seq_pipe.set_hf_cache_dir_override(FLAGS.hf_cache_dir)

        # import new modules
        if FLAGS.module_import:
            for module in FLAGS.module_import:
                importlib.import_module(module)

        # get task
        task = seq_pipe.get_task(FLAGS.task)

        # To prevent other processes from generating data redundantly,
        # if the local rank is 0, data is created before init_process_group.
        get_dataset(task, split=FLAGS.train_split)

        # download pretrained model if there is no cache
        model_class = loader.load_model(FLAGS.model_name)
        model_kwargs = task.additional_task_info
        model = model_class.from_pretrained(
            FLAGS.pre_trained_model, **model_kwargs)

    # if the world size is bigger than 1, init process group(sync)
    if FLAGS.distributed:
        FLAGS.gpu = FLAGS.local_rank
        torch.cuda.set_device(FLAGS.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        FLAGS.world_size = torch.distributed.get_world_size()

    # w/o main proc
    if not FLAGS.distributed or FLAGS.local_rank != 0:
        # parsing and binding gin configs
        gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

        # import tasks after binding gin configs
        # pretrained model will be downloaded if there is no cache
        # downloading pretrained model is only required once,
        # this code will be excuted on the main proc in the distributed setting
        from ke_t5.task import task

        # override data_dir and cache dir for huggingface datasets
        seq_pipe.set_hf_data_dir_override(FLAGS.hf_data_dir)
        seq_pipe.set_hf_cache_dir_override(FLAGS.hf_cache_dir)

        # import new modules
        if FLAGS.module_import:
            for module in FLAGS.module_import:
                importlib.import_module(module)

        # get task
        task = seq_pipe.get_task(FLAGS.task)

    metric_meter = utils.MetricMeter(task)
    metric_meter.add_average_meter("loss")

    # only for main processors
    summary_logger = None

    if FLAGS.local_rank == 0 or not FLAGS.distributed:
        # best_function
        best_fn = task.best_fn
        best_score = best_fn.get_min_score()

        # create directory
        path_info = utils.create_directory_info(FLAGS)

        # create summary_logger
        summary_logger = utils.TensorboardXLogging(path_info["logs_dir"])
    
    path_info = utils.create_directory_info(FLAGS, create_dir=False)

    # get model
    if not FLAGS.distributed or FLAGS.local_rank != 0:
        model_class = loader.load_model(FLAGS.model_name)
        model_kwargs = task.additional_task_info
        model = model_class.from_pretrained(
            FLAGS.pre_trained_model, **model_kwargs)
    model = model.cuda()

    # get optimizer
    optimizer_cls = get_optimizer()
    optimizer = optimizer_cls(model.parameters())

    # wrap model using DDP
    if FLAGS.distributed:
        model = DDP(model,
                    device_ids=[FLAGS.local_rank],
                    output_device=FLAGS.local_rank)

    if FLAGS.test:
        test_dataset = get_dataset(task, split=FLAGS.valid_split)
        if FLAGS.pass_only_model_io:
            test_dataset.set_format('torch', columns=task.model_input_columns, device='cuda')
        else:
            test_dataset.set_format('torch', columns=task.model_input_columns, device='cuda', output_all_columns=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=FLAGS.batch_size, 
            shuffle=False, 
            num_workers=FLAGS.workers,
            sampler=test_sampler,
            collate_fn=utils.collate_variable_length)
        metric_meter = validate(test_loader, model, 0, FLAGS, task, metric_meter)
        if FLAGS.local_rank == 0 or not FLAGS.distributed:
            score_log = metric_meter.get_score_str("test")
            logging.info('\n' + '-'*10 + 'test'+'-'*10+'\n'+score_log+ '\n' + '-'*24)
        exit()

    # load dataset
    train_dataset = get_dataset(task, split=FLAGS.train_split)
    test_dataset = get_dataset(task, split=FLAGS.valid_split)

    # set dataset as pytorch dataset
    if FLAGS.pass_only_model_io:
        train_dataset.set_format('torch', columns=task.model_input_columns, device='cuda')
        test_dataset.set_format('torch', columns=task.model_input_columns, device='cuda')
    else:
        train_dataset.set_format('torch', columns=task.model_input_columns, device='cuda', output_all_columns=True)
        test_dataset.set_format('torch', columns=task.model_input_columns, device='cuda', output_all_columns=True)

    # create sampler for distributed data loading without redundant
    train_sampler = None
    test_sampler = None
    if FLAGS.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)

    # create data loader
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=FLAGS.workers,
                              sampler=train_sampler,
                              collate_fn=utils.collate_variable_length)
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             num_workers=FLAGS.workers,
                             sampler=test_sampler,
                             collate_fn=utils.collate_variable_length)

    scheduler = None
    # learning rate scheduler
    if FLAGS.schedule:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=FLAGS.learning_rate,
            epochs=FLAGS.epochs,
            last_epoch=-1,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="linear"
        )
    
    if FLAGS.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(FLAGS.resume):
                logging.info("=> loading checkpoint '{}'".format(FLAGS.resume))
                checkpoint = torch.load(
                    FLAGS.resume, map_location=lambda storage, loc: storage.cuda(FLAGS.gpu))
                FLAGS.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if scheduler is not None and "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])

                if FLAGS.local_rank == 0 or not FLAGS.distributed:
                    best_score = checkpoint['best_score']
                logging.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(FLAGS.resume, checkpoint['epoch']))
            elif FLAGS.resume.lower()=='true':
                FLAGS.resume = path_info['ckpt_path']
                resume()
            elif FLAGS.resume.lower()=='best':
                FLAGS.resume = path_info['best_model_path']
                resume()
            else:
                logging.info("=> no checkpoint found at '{}'".format(FLAGS.resume))
        resume()

    if FLAGS.hf_path:
        if FLAGS.hf_path.lower() == 'default':
            FLAGS.hf_path = os.path.join(path_info['model_dir'], "hf")
        if FLAGS.local_rank == 0 and FLAGS.distributed:
            model.module.save_pretrained(FLAGS.hf_path)
            logging.info('hf model is saved in {}'.format(FLAGS.hf_path))
        elif not FLAGS.distributed:
            model.save_pretrained(FLAGS.hf_path)
            logging.info('hf model is saved in {}'.format(FLAGS.hf_path))
        exit()


    # run training
    for epoch in range(FLAGS.start_epoch, FLAGS.epochs):
        # set epoch to train sampler 
        # for different order of example between epochs
        if FLAGS.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, epoch,
              FLAGS, task, metric_meter, scheduler, summary_logger)

        metric_meter = validate(test_loader, model, epoch, FLAGS, task, metric_meter)
        if FLAGS.local_rank == 0 or not FLAGS.distributed:
            avg_scores = metric_meter.get_average_scores()

            is_best, best_score = best_fn.is_best(avg_scores, best_score)

            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }, is_best,
                path_info["ckpt_path"],
                path_info["best_model_path"])

            summary_logger(
                avg_scores,
                epoch,
                FLAGS.task,
                "eval")


def validate(eval_loader, model, epoch, args, task, metric_meter):
    batch_time = utils.AverageMeter()
    
    # reset metric_meter
    metric_meter.reset()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for step_inbatch, batch in enumerate(eval_loader):
            # select model inputs
            inputs = task.select_model_inputs(batch)
            # forward pass
            outputs = model(
                **inputs
            )

            # get loss and logits
            loss = outputs[0]
            logits = outputs[1]

            # get predictions
            if task.logit_to_id and isinstance(logits, torch.Tensor):
                predictions = utils.get_ids_from_logits(logits.clone())
            elif isinstance(logits, torch.Tensor):
                predictions = logits.clone()
            else:
                predictions = logits

            # update metrics
            metric_meter.update_scores("loss", {'score': loss.cpu().numpy(), 'count': 1})
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            gathered_dict = {k:v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
            gathered_dict['predictions'] = predictions
            metric_meter.update_metrics(gathered_dict)

            # reduce average scores
            average_scores = metric_meter.get_average_scores()
            if args.distributed:
                average_scores = {
                    k:{
                        'score': reduce_sum_tensor(torch.tensor(v['score']*v['count'], device='cuda')).cpu().numpy(), 
                        'count': reduce_sum_tensor(torch.tensor(v['count'], device='cuda')).cpu().numpy()
                        } for k, v in average_scores.items()
                }
                average_scores = {k:{'score': v['score']/v['count'], 'count': v['count']} for k, v in average_scores.items()}

            if step_inbatch % args.print_freq == 0:
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                if args.local_rank == 0 or not args.distributed:
                    score_log = metric_meter.get_score_str("eval", average_scores=average_scores)

                    logging.info('-----Evaluation----- \nEpoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Speed {3:.3f} ({4:.3f})\t'.format(
                                epoch, step_inbatch, len(eval_loader),
                                args.batch_size/batch_time.val,
                                args.batch_size/batch_time.avg,
                                batch_time=batch_time) + score_log)
    
    # recude final scores
    average_scores = metric_meter.get_average_scores()
    if args.distributed:
        average_scores = {
                    k:{
                        'score': reduce_sum_tensor(torch.tensor(v['score']*v['count'], device='cuda')).cpu().numpy(), 
                        'count': reduce_sum_tensor(torch.tensor(v['count'], device='cuda')).cpu().numpy()
                        } for k, v in average_scores.items()
                }
        average_scores = {k:{'score': v['score']/v['count'], 'count': v['count']} for k, v in average_scores.items()}

    if args.local_rank == 0 or not args.distributed:
        metric_meter.reset()
        metric_meter.set_average_scores(average_scores)
        score_log = metric_meter.get_score_str("eval", average_scores=average_scores)
        logging.info('-----Evaluation-----\n' + score_log)

    return metric_meter


def train(train_loader, model, optimizer, epoch, args, task, metric_meter=None, scheduler=None, summary_logger=None):
    # calc batch time
    batch_time = utils.AverageMeter()
    
    # reset metric_meter
    metric_meter.reset()
    steps_per_epoch = len(train_loader)

    # switch to train mode
    model.train()
    end = time.time()

    for step_inbatch, batch in enumerate(train_loader):
        # select model inputs
        inputs = task.select_model_inputs(batch)

        # forward pass
        outputs = model(
            **inputs
        )

        # get loss and logits
        loss = outputs[0]
        logits = outputs[1]

        # schedule learning rate
        if scheduler is not None:
            scheduler.step()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get predictions
        if task.logit_to_id and isinstance(logits, torch.Tensor):
            predictions = utils.get_ids_from_logits(logits.clone())
        elif isinstance(logits, torch.Tensor):
            predictions = logits.clone()
        else:
            predictions = logits

        global_step = epoch*steps_per_epoch + step_inbatch
        if global_step % args.print_freq == 0:
            with torch.no_grad():
                batch_time.update((time.time() - end)/args.print_freq)
                end = time.time()

                # update metrics
                metric_meter.update_scores("loss", {'score': loss.cpu().numpy(), 'count': 1})
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                gathered_dict = {k:v.cpu().numpy() if torch.is_tensor(v) else v for k, v in batch.items()}
                gathered_dict['predictions'] = predictions
                
                metric_meter.update_metrics(gathered_dict)

                average_scores = metric_meter.get_average_scores()
                if args.distributed:
                    average_scores = {
                        k:{
                            'score': reduce_sum_tensor(torch.tensor(v['score']*v['count'], device='cuda')).cpu().numpy(), 
                            'count': reduce_sum_tensor(torch.tensor(v['count'], device='cuda')).cpu().numpy()
                            } for k, v in average_scores.items()
                    }
                    average_scores = {k:{'score': v['score']/v['count'], 'count': v['count']} for k, v in average_scores.items()}

                if args.local_rank == 0 or not args.distributed:
                    score_log = summary_logger(
                        average_scores,
                        global_step,
                        args.task,
                        "train")
                    logging.info('-----Training----- \nEpoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Speed {3:.3f} ({4:.3f})\t'.format(
                              epoch, step_inbatch, steps_per_epoch,
                              args.batch_size/batch_time.val,
                              args.batch_size/batch_time.avg,
                              batch_time=batch_time)+score_log)



def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

if __name__ == "__main__":
    app.run(main)


# python -m torch.distributed.launch --nproc_per_node=1 train_ddp.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="train.gin" --model_name="transformers:AutoModelForSequenceClassification" --pre_trained_model="bert-base-uncased"
# python -m torch.distributed.launch --nproc_per_node=1 train_ddp.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_param="get_dataset.sequence_length={'inputs':512, 'targets':512}" --model_name="transformers:AutoModelForSequenceClassification" --pre_trained_model="bert-base-uncased"

# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="train.gin"
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" --gin_file="train.gin" --resume output/ke_t5.models.models:T5EncoderForSequenceClassificationMean_KETI-AIR/ke-t5-small/weights/best_model.pth --test true --valid_split "test"


# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --task 'klue_tc'
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --task 'klue_re'
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --task 'klue_nli'

# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --model_name transformers:T5ForConditionalGeneration --task 'klue_nli_gen'
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --model_name transformers:T5ForConditionalGeneration --task 'nikl_summarization_topic'


# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --model_name transformers:T5ForConditionalGeneration --task 'korquad_gen'
# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --model_name ke_t5.models.models:T5EncoderForSequenceClassificationMean --task 'klue_tc'

# python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --gin_file="train.gin" --model_name transformers:T5ForConditionalGeneration --task 'klue_tc_gen'

