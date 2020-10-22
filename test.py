from __future__ import division
from setproctitle import setproctitle as ptitle

import os
import time
import torch
import logging
import numpy as np
from tensorboardX import SummaryWriter

from model import build_model
from utils import setup_logger
from player_util import Agent
from environment import create_env


def test(args, shared_model, optimizer, train_modes, n_iters):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}/logger'.format(args.log_dir))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env, args)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(player.env.observation_space, player.env.action_space, args, device).to(device)
    player.model.eval()
    max_score = -100

    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        len_sum = 0
        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            while True:
                if args.render:
                    env.render()
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                rotation_sum_ep += player.rotation
                if player.done:
                    AG += reward_sum_ep[0]/rotation_sum_ep*player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time()-t0)
                    n_iter = 0
                    for n in n_iters:
                        n_iter += n

                    for i, r_i in enumerate(reward_sum_ep):
                        writer.add_scalar('test/reward'+str(i), r_i, n_iter)

                    fps_all.append(fps)
                    writer.add_scalar('test/fps', fps, n_iter)
                    writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break

        # player.max_length:
        ave_AG = AG/args.test_eps
        ave_reward_sum = reward_sum/args.test_eps
        len_mean = len_sum/args.test_eps
        reward_step = reward_sum / len_sum
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)

        log['{}_log'.format(args.env)].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, AG {7}".
            format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)
            ))

        # save model
        if ave_reward_sum[0] >= max_score:
            print('save best!')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'best.pth')
        else:
            model_dir = os.path.join(args.log_dir, 'new.pth'.format(args.env))
        state_to_save = {"model": player.model.state_dict(),
                         "optimizer": optimizer.state_dict()}
        torch.save(state_to_save, model_dir)

        time.sleep(args.sleep_time)
        if n_iter > args.max_step:
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            break
