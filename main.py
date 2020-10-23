from __future__ import print_function, division
import os
import time
import torch
import argparse
from datetime import datetime
import torch.multiprocessing as mp

from test import test
from train import train
from model import build_model
from environment import create_env
from shared_optim import SharedRMSprop, SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T', help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy', type=float, default=0.01, metavar='T', help='parameter for entropy (default: 0.01)')
parser.add_argument('--grad-entropy', type=float, default=1.0, metavar='T', help='parameter for entropy (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, default=1, metavar='W', help='how many training processes to use (default: 32)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in A3C (default: 300)')
parser.add_argument('--test-eps', type=int, default=2, metavar='M', help='maximum length of an episode (default: 10000)')
parser.add_argument('--env', default='simple', metavar='Pose-v0', help='environment to train on (default: Pose-v0|Pose-v1)')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-coordinator-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--load-executor-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='single', metavar='M', help='multi-shapleyV|')
parser.add_argument('--gpu-ids', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--norm-reward', dest='norm_reward', action='store_true', help='normalize image')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--fix', dest='fix', action='store_true', help='fix random seed')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use an optimizer without shared statistics.')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='his')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--lstm-out', type=int, default=128, metavar='LO', help='lstm output size')
parser.add_argument('--sleep-time', type=int, default=0, metavar='LO', help='seconds')
parser.add_argument('--max-step', type=int, default=5000000, metavar='LO', help='max learning steps')
parser.add_argument('--render_save', dest='render_save', action='store_true', help='render save')

def start():
    args = parser.parse_args()
    args.shared_optimizer = True
    if args.gpu_ids == -1:
        torch.manual_seed(args.seed)
        args.gpu_ids = [-1]
        device_share = torch.device('cpu')
        mp.set_start_method('spawn')
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))
    env = create_env(args.env, args)
    shared_model = build_model(env.observation_space, env.action_space, args, device_share).to(device_share)
    shared_model.share_memory()
    env.close()
    del env

    if args.load_coordinator_dir is not None:
        saved_state = torch.load(
            args.load_coordinator_dir,
            map_location=lambda storage, loc: storage)
        if args.load_coordinator_dir[-3:] == 'pth':
            shared_model.load_state_dict(saved_state['model'], strict=False)
        else:
            shared_model.load_state_dict(saved_state)

    params = shared_model.parameters()
    if args.shared_optimizer:
        print('share memory')
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(params, lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)

    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()

    p = mp.Process(target=test, args=(args, shared_model, optimizer, train_modes, n_iters))
    p.start()
    processes.append(p)
    time.sleep(args.sleep_time)

    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer, train_modes, n_iters))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    for p in processes:
        time.sleep(args.sleep_time)
        p.join()


if __name__=='__main__':
    start()

