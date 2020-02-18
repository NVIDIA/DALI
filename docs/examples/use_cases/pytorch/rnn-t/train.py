# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import itertools
import os
import time
import toml
import torch
import apex
from apex import amp
import random
import numpy as np
import math
from dataset import AudioToTextDataLayer
from helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch, Optimization, add_blank_label, AmpOptimizations, model_multi_gpu, print_dict, print_once
from model_rnnt import RNNT
from decoders import RNNTGreedyDecoder
from loss import RNNTLoss
from optimizers import Novograd, AdamW

import torchvision

from tb_logger import DummyLogger, TensorBoardLogger
import preprocessing
import dali


def lr_decay(N, step, learning_rate):
    """
    learning rate decay
    Args:
        learning_rate: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
    """
    min_lr = 0.00001
    res = learning_rate * ((N - step) / N) ** 2
    return max(res, min_lr)

def lr_warmup(warmup_steps, step, learning_rate):
    return min(1, (step / warmup_steps)) * learning_rate

def save(model, optimizer, epoch, output_dir):
    """
    Saves model checkpoint
    Args:
        model: model
        optimizer: optimizer
        epoch: epoch of model training
        output_dir: path to save model checkpoint
    """
    class_name = model.__class__.__name__
    unix_time = time.time()
    file_name = "{0}_{1}-epoch-{2}.pt".format(class_name, unix_time, epoch)
    print_once("Saving module {0} in {1}".format(class_name, os.path.join(output_dir, file_name)))
    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        save_checkpoint={
                        'epoch': epoch,
                        'state_dict': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }

        torch.save(save_checkpoint, os.path.join(output_dir, file_name))
    print_once('Saved.')


def evaluator(model, data_transforms, loss_fn, greedy_decoder, labels, eval_datasets, logger):
    """Evaluates model on evaluation dataset
    """

    def evalutaion(epoch=0):
        model.eval()

        for dataset, frequency, name in eval_datasets:
            if epoch % frequency != 0:
                continue

            print_once(f"Doing {name} ....................... ......  ... .. . .")

            with torch.no_grad():
                _global_var_dict = {
                    'EvalLoss': [],
                    'predictions': [],
                    'transcripts': [],
                }
                dataloader = dataset.data_iterator
                for data in dataloader:

                    t_audio_signal_e, t_a_sig_length_e, t_transcript_e, t_transcript_len_e = data_transforms(data)

                    t_log_probs_e, (x_len, y_len) = model(
                        ((t_audio_signal_e, t_transcript_e), (t_a_sig_length_e, t_transcript_len_e)),
                    )
                    t_loss_e = loss_fn(
                        (t_log_probs_e, x_len), (t_transcript_e, y_len)
                    )
                    del t_log_probs_e

                    t_predictions_e = greedy_decoder.decode(t_audio_signal_e, t_a_sig_length_e)

                    values_dict = dict(
                        loss=[t_loss_e],
                        predictions=[t_predictions_e],
                        transcript=[t_transcript_e],
                        transcript_length=[t_transcript_len_e]
                    )
                    process_evaluation_batch(values_dict, _global_var_dict, labels=labels)

                # final aggregation across all workers and minibatches) and logging of results
                wer, eloss = process_evaluation_epoch(_global_var_dict)
                logger.log_scalar('loss', eloss, epoch, name)
                logger.log_scalar('wer', wer, epoch, name)

                print_once(f"==========>>>>>>{name} Loss: {eloss}\n")
                print_once(f"==========>>>>>>{name} WER: {wer}\n")

    return evalutaion


def train(
        data_layer,
        model,
        loss_fn,
        greedy_decoder,
        optimizer,
        optim_level,
        labels,
        multi_gpu,
        data_transforms,
        args,
        evalutaion,
        use_dali,
        logger,
        fn_lr_policy):
    """Trains model
    Args:
        data_layer: training data layer
        model: model ( encapsulates data processing, encoder, decoder)
        loss_fn: loss function
        greedy_decoder: greedy ctc decoder
        optimizer: optimizer
        optim_level: AMP optimization level
        labels: list of output labels
        multi_gpu: true if multi gpu training
        args: script input argument list
        fn_lr_policy: function returning lr in given step
    """
    print_once("Starting .....")
    start_time = time.time()

    train_dataloader = data_layer.data_iterator
    epoch = args.start_epoch
    step = epoch * args.step_per_epoch

    while True:
        if multi_gpu:
            data_layer.sampler.set_epoch(epoch)
        print_once("Starting epoch {0}, step {1}".format(epoch, step))
        last_epoch_start = time.time()
        batch_counter = 0
        average_loss = 0
        for data in train_dataloader:

            if batch_counter == 0:

                adjusted_lr = fn_lr_policy(step)
                for param_group in optimizer.param_groups:
                        param_group['lr'] = adjusted_lr
                optimizer.zero_grad()
                last_iter_start = time.time()

            t_audio_signal_t, \
            t_a_sig_length_t, \
            t_transcript_t, \
            t_transcript_len_t = data if use_dali else data_transforms(data)

            # TODO(mszolucha) figure out, if this can be done inside DALI
            if use_dali:
                t_audio_signal_t = t_audio_signal_t.permute(2, 0, 1)

            model.train()

            t_log_probs_t, (x_len, y_len) = model(
                ((t_audio_signal_t, t_transcript_t), (t_a_sig_length_t, t_transcript_len_t)),
            )

            t_loss_t = loss_fn(
                (t_log_probs_t, x_len), (t_transcript_t, y_len)
            )
            logger.log_scalar('loss', t_loss_t.item(), step)
            del t_log_probs_t
            if args.gradient_accumulation_steps > 1:
                t_loss_t = t_loss_t / args.gradient_accumulation_steps

            if optim_level in AmpOptimizations:
                with amp.scale_loss(t_loss_t, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                t_loss_t.backward()
            batch_counter += 1
            average_loss += t_loss_t.item()

            if batch_counter % args.gradient_accumulation_steps == 0:
                optimizer.step()

                if (step + 1) % args.train_frequency == 0:
                    t_predictions_t = greedy_decoder.decode(t_audio_signal_t, t_a_sig_length_t)

                    e_tensors = [t_predictions_t, t_transcript_t, t_transcript_len_t]
                    train_wer = monitor_asr_train_progress(e_tensors, labels=labels)
                    print_once("Loss@Step: {0}  ::::::: {1}".format(step, str(average_loss)))
                    print_once("Step time: {0} seconds".format(time.time() - last_iter_start))
                    logger.log_scalar('wer', train_wer, step)

                step += 1
                batch_counter = 0
                average_loss = 0
                if args.num_steps is not None and step >= args.num_steps:
                    break

        evalutaion(epoch)

        if args.num_steps is not None and step >= args.num_steps:
            break
        print_once("Finished epoch {0} in {1}".format(epoch, time.time() - last_epoch_start))
        epoch += 1
        if epoch % args.save_frequency == 0 and epoch > 0:
            save(model, optimizer, epoch, output_dir=args.output_dir)
        if args.num_steps is None and epoch >= args.num_epochs:
            break
    print_once("Done in {0}".format(time.time() - start_time))
    print_once("Final Evaluation ....................... ......  ... .. . .")
    evalutaion()
    save(model, optimizer, epoch, output_dir=args.output_dir)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = args.cudnn

    args.local_rank = os.environ.get('LOCAL_RANK', args.local_rank)
    # set up distributed training
    if args.local_rank is not None:
        args.local_rank = int(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        print_once("DISTRIBUTED TRAINING with {} gpus".format(torch.distributed.get_world_size()))

    # define amp optimiation level
    if args.fp16:
        optim_level = Optimization.mxprO1
    else:
        optim_level = Optimization.mxprO0

    model_definition = toml.load(args.model_toml)
    dataset_vocab = model_definition['labels']['labels']
    ctc_vocab = add_blank_label(dataset_vocab)

    train_manifest = args.train_manifest
    val_manifest = args.val_manifest
    tst_manifest = args.tst_manifest
    featurizer_config = model_definition['input']
    featurizer_config_eval = model_definition['input_eval']
    featurizer_config["optimization_level"] = optim_level
    featurizer_config_eval["optimization_level"] = optim_level

    sampler_type = featurizer_config.get("sampler", 'default')
    perturb_config = model_definition.get('perturb', None)
    if args.pad_to_max:
        assert(args.max_duration > 0)
        featurizer_config['max_duration'] = args.max_duration
        featurizer_config_eval['max_duration'] = args.max_duration
        featurizer_config['pad_to'] = "max"
        featurizer_config_eval['pad_to'] = "max"
    print_once('model_config')
    print_dict(model_definition)

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient accumulation steps parameter {}'.format(args.gradient_accumulation_steps))
    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('gradient accumulation step {} is not divisible by batch size {}'.format(args.gradient_accumulation_steps, args.batch_size))


    preprocessor = preprocessing.AudioPreprocessing(**featurizer_config)
    preprocessor.cuda()

    augmentations = preprocessing.SpectrogramAugmentation(**featurizer_config)
    augmentations.cuda()

    train_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cuda() for x in xs],
        lambda xs: [*preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [augmentations(xs[0]),   *xs[1:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    eval_transforms = torchvision.transforms.Compose([
        lambda xs: [x.cuda() for x in xs],
        lambda xs: [*preprocessor(xs[0:2]), *xs[2:]],
        lambda xs: [xs[0].permute(2, 0, 1), *xs[1:]],
    ])

    data_layer = AudioToTextDataLayer(
                                    dataset_dir=args.dataset_dir,
                                    featurizer_config=featurizer_config,
                                    perturb_config=perturb_config,
                                    manifest_filepath=train_manifest,
                                    labels=dataset_vocab,
                                    batch_size=args.batch_size // args.gradient_accumulation_steps,
                                    multi_gpu=multi_gpu,
                                    pad_to_max=args.pad_to_max,
                                    sampler=sampler_type)

    if args.use_dali:
        mfest = train_manifest.split(',')  #TODO
        dali_data_layer = dali.DaliDataLayer(dataset_path=args.dataset_dir, json_name=mfest[0],
                                             batch_size=args.batch_size // args.gradient_accumulation_steps, ngpus=1)

    eval_datasets = [(
        AudioToTextDataLayer(
            dataset_dir=args.dataset_dir,
            featurizer_config=featurizer_config_eval,
            manifest_filepath=val_manifest,
            labels=dataset_vocab,
            batch_size=args.eval_batch_size,
            multi_gpu=multi_gpu,
            pad_to_max=args.pad_to_max
        ),
        args.eval_frequency,
        'Eval clean',
    )]

    if tst_manifest:
        eval_datasets.append((
            AudioToTextDataLayer(
                dataset_dir=args.dataset_dir,
                featurizer_config=featurizer_config_eval,
                manifest_filepath=tst_manifest,
                labels=dataset_vocab,
                batch_size=args.eval_batch_size,
                multi_gpu=multi_gpu,
                pad_to_max=args.pad_to_max
            ),
            args.test_frequency,
            'Test other',
        ))

    model = RNNT(
        feature_config=featurizer_config,
        rnnt=model_definition['rnnt'],
        num_classes=len(ctc_vocab)
    )

    if args.ckpt is not None:
        print_once("loading model from {}".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        args.start_epoch = checkpoint['epoch']
    else:
        args.start_epoch = 0

    loss_fn = RNNTLoss(blank=len(ctc_vocab) - 1)

    N = len(data_layer)
    if sampler_type == 'default':
        args.step_per_epoch = math.ceil(N / (args.batch_size * (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())))
    elif sampler_type == 'bucket':
        args.step_per_epoch = int(len(data_layer.sampler) / args.batch_size )

    print_once('-----------------')
    print_once('Have {0} examples to train on.'.format(N))
    print_once('Have {0} steps / (gpu * epoch).'.format(args.step_per_epoch))
    print_once('-----------------')

    constant_lr_policy = lambda _: args.lr
    fn_lr_policy = constant_lr_policy
    if args.lr_decay:
        pre_decay_policy = fn_lr_policy
        fn_lr_policy = lambda s: lr_decay(args.num_epochs * args.step_per_epoch, s, pre_decay_policy(s))
    if args.lr_warmup:
        pre_warmup_policy = fn_lr_policy
        fn_lr_policy = lambda s: lr_warmup(args.lr_warmup, s, pre_warmup_policy(s) )


    model.cuda()


    if args.optimizer_kind == "novograd":
        optimizer = Novograd(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer_kind == "adam":
        optimizer = AdamW(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("invalid optimizer choice: {}".format(args.optimizer_kind))

    if optim_level in AmpOptimizations:
        model, optimizer = amp.initialize(
            min_loss_scale=0.125,
            models=model,
            optimizers=optimizer,
            opt_level=AmpOptimizations[optim_level]
        )

    if args.ckpt is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model = model_multi_gpu(model, multi_gpu)
    print_once(model)
    print_once("# parameters: {}".format(sum(p.numel() for p in model.parameters())))
    greedy_decoder = RNNTGreedyDecoder(len(ctc_vocab) - 1, model.module if multi_gpu else model)

    if args.tb_path and args.local_rank == 0:
        logger = TensorBoardLogger(args.tb_path, model.module if multi_gpu else model, args.histogram)
    else:
        logger = DummyLogger()

    train(
        data_layer=data_layer if not args.use_dali else dali_data_layer,
        model=model,
        loss_fn=loss_fn,
        greedy_decoder=greedy_decoder,
        optimizer=optimizer,
        data_transforms=train_transforms,
        labels=ctc_vocab,
        optim_level=optim_level,
        multi_gpu=multi_gpu,
        fn_lr_policy=fn_lr_policy,
        evalutaion=evaluator(model, eval_transforms, loss_fn, greedy_decoder, ctc_vocab, eval_datasets, logger),
        use_dali=args.use_dali,
        logger=logger,
        args=args)

def parse_args():
    parser = argparse.ArgumentParser(description='RNNT Training Reference')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int, help='data batch size')
    parser.add_argument("--eval_batch_size", default=1, type=int, help='eval data batch size')
    parser.add_argument("--num_epochs", default=10, type=int, help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--num_steps", default=None, type=int, help='if specified overwrites num_epochs and will only train for this number of iterations')
    parser.add_argument("--save_freq", dest="save_frequency", default=300, type=int, help='number of epochs until saving checkpoint. will save at the end of training too.')
    parser.add_argument("--eval_freq", dest="eval_frequency", default=1, type=int, help='number of epochs until doing evaluation on full dataset')
    parser.add_argument("--test_freq", dest="test_frequency", default=2, type=int, help='number of epochs until doing test on full dataset')
    parser.add_argument("--train_freq", dest="train_frequency", default=25, type=int, help='number of iterations until printing training statistics on the past iteration')
    parser.add_argument("--lr", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='weight decay rate')
    parser.add_argument("--train_manifest", type=str, required=True, help='relative path given dataset folder of training manifest file')
    parser.add_argument("--model_toml", type=str, required=True, help='relative path given dataset folder of model configuration file')
    parser.add_argument("--val_manifest", type=str, required=True, help='relative path given dataset folder of evaluation manifest file')
    parser.add_argument("--tst_manifest", type=str, required=False, help='relative path given dataset folder of test manifest file')
    parser.add_argument("--max_duration", type=float, help='maximum duration of audio samples for training and evaluation')
    parser.add_argument("--pad_to_max", action="store_true", default=False, help="pad sequence to max_duration")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help='number of accumulation steps')
    parser.add_argument("--optimizer", dest="optimizer_kind", default="novograd", type=str, help='optimizer')
    parser.add_argument("--dataset_dir", dest="dataset_dir", required=True, type=str, help='root dir of dataset')
    parser.add_argument("--lr_decay", action="store_true", default=False, help='use learning rate decay')
    parser.add_argument("--lr_warmup", type=int, default=None, help='if provided, the learning rate will linearly scale for given number of iterations from zero')
    parser.add_argument("--cudnn", action="store_true", default=False, help="enable cudnn benchmark")
    parser.add_argument("--fp16", action="store_true", default=False, help="use mixed precision training")
    parser.add_argument("--use_dali", action="store_true", default=False, help="use DALI for data preprocessing pipeline")
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--ckpt", default=None, type=str, help="if specified continues training from given checkpoint. Otherwise starts from beginning")
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--tb_path", default=None, type=str, help='where to store tensorboard data')
    parser.add_argument("--histogram", default=False, action='store_true', help='whether to log param and grad histograms')
    args=parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)
