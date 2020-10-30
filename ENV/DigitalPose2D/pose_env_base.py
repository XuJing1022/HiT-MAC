import os
import json
import math
import torch
import random
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from datetime import datetime

from ENV.DigitalPose2D.render import render
from model import A3C_Single
from utils import goal_id_filter
from main import parser

args = parser.parse_args()


class Pose_Env_Base:
    def __init__(self, reset_type,
                 nav='Goal',  # Random, Goal
                 config_path="PoseEnvLarge_multi.json",
                 render_save=False,
                 setting_path=None,
                 slave_rule=args.load_executor_dir is None
                 ):

        self.nav = nav
        self.reset_type = reset_type
        self.ENV_PATH = 'ENV/DigitalPose2D'

        if setting_path:
            self.SETTING_PATH = setting_path
        else:
            self.SETTING_PATH = os.path.join(self.ENV_PATH, config_path)
        with open(self.SETTING_PATH, encoding='utf-8') as f:
            setting = json.load(f)

        self.env_name = setting['env_name']

        self.cam_id = setting['cam_id']
        self.n = len(self.cam_id)
        self.cam_area = np.array(setting['cam_area'])

        self.target_list = setting['targets']
        self.num_target = len(self.target_list)
        self.reset_area = setting['reset_area']

        # for executor
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions_player = setting['continous_actions_player']

        self.max_steps = setting['max_steps']

        self.visual_distance = setting['visual_distance']
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 1000)

        # define action space for coordinator
        self.action_space = [spaces.Discrete(2) for i in range(self.n * self.num_target)]
        self.rotation_scale = setting['rotation_scale']

        # define observation space
        self.state_dim = 4
        self.observation_space = np.zeros((self.n, self.num_target, self.state_dim), int)

        self.render_save = render_save
        self.render = args.render

        self.cam = dict()
        for i in range(len(self.cam_id) + 1):
            self.cam[i] = dict(
                location=[0, 0],
                rotation=[0],
            )

        self.count_steps = 0
        self.goal_keep = 0
        self.KEEP = 10
        self.goals4cam = np.ones([self.n, self.num_target])

        # construct target_agent
        if 'Goal' in self.nav:
            self.random_agents = [GoalNavAgent(i, self.continous_actions_player, self.reset_area)
                                  for i in range(self.num_target)]

        self.slave_rule = slave_rule
        if not self.slave_rule:
            self.device = torch.device('cpu')
            self.slave = A3C_Single(np.zeros((1, 1, 4)), [spaces.Discrete(3)], args)
            self.slave = self.slave.to(self.device)
            saved_state = torch.load(
                args.load_executor_dir,  # 'trainedModel/best_executor.pth',
                map_location=lambda storage, loc: storage)
            self.slave.load_state_dict(saved_state['model'], strict=False)
            self.slave.eval()

    def set_location(self, cam_id, loc):
        self.cam[cam_id]['location'] = loc

    def get_location(self, cam_id):
        return self.cam[cam_id]['location']

    def set_rotation(self, cam_id, rot):
        for i in range(len(rot)):
            if rot[i] > 180:
                rot[i] -= 360
            if rot[i] < -180:
                rot[i] += 360
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation']

    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d

    def reset(self):

        # reset targets
        self.target_pos_list = np.array([[
            float(np.random.randint(self.start_area[0], self.start_area[1])),
            float(np.random.randint(self.start_area[2], self.start_area[3]))] for _ in range(self.num_target)])
        # reset agent
        for i in range(len(self.random_agents)):
            if 'Goal' in self.nav:
                self.random_agents[i].reset()

        # reset camera
        camera_id_list = [i for i in self.cam_id]
        random.shuffle(camera_id_list)

        for i, cam in enumerate(self.cam_id):
            cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                       np.random.randint(self.cam_area[i][2], self.cam_area[i][3])
                       ]
            self.set_location(camera_id_list[i], cam_loc)  # shuffle

        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)

            # start with non-focusing
            angle_h = np.random.randint(-180, 180)
            cam_rot[0] += angle_h * 1.0

            self.set_rotation(cam, cam_rot)

        self.count_steps = 0
        self.goal_keep = 0
        self.goals4cam = np.ones([self.n, self.num_target])

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
        )

        gt_directions = []
        gt_distance = []
        cam_info = []
        for i, cam in enumerate(self.cam_id):
            # for target navigation
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            cam_info.append([cam_loc, cam_rot])
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(cam_loc + cam_rot)

        info['Directions'] = np.array(gt_directions)
        info['Distance'] = np.array(gt_distance)
        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Reward'], info['Global_reward'], others = self.multi_reward(cam_info, self.goals4cam)
        if others:
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']

        state, self.state_dim = self.preprocess_pose(info)
        return state

    def target_move(self):
        step = 10
        if 'Random' in self.nav:
            for i in range(self.num_target):
                self.target_pos_list[i][:3] += [np.random.randint(-1 * step, step),
                                                np.random.randint(-1 * step, step)]
        elif 'Goal' in self.nav:
            delta_time = 0.13
            for i in range(self.num_target):  # only one
                loc = list(self.target_pos_list[i])
                action = self.random_agents[i].act(loc)

                target_hpr_now = np.array(action[1:])
                delta_x = target_hpr_now[0] * action[0] * delta_time
                delta_y = target_hpr_now[1] * action[0] * delta_time
                while loc[0] + delta_x < self.reset_area[0] or loc[0] + delta_x > self.reset_area[1] or \
                        loc[1] + delta_y < self.reset_area[2] or loc[1] + delta_y > self.reset_area[3]:
                    action = self.random_agents[i].act(loc)

                    target_hpr_now = np.array(action[1:])
                    delta_x = target_hpr_now[0] * action[0] * delta_time
                    delta_y = target_hpr_now[1] * action[0] * delta_time

                self.target_pos_list[i][0] += delta_x
                self.target_pos_list[i][1] += delta_y

    def step(self, actions):

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps
        )

        self.goals4cam = np.squeeze(actions)

        # target move
        self.target_move()

        # camera move
        cam_info = []
        for i, cam in enumerate(self.cam_id):
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            cam_info.append([cam_loc, cam_rot])
        r, gr, others, cam_info = self.simulate(self.goals4cam, cam_info, keep=10)
        for i, cam in enumerate(self.cam_id):
            cam_loc, cam_rot = cam_info[i]
            self.set_rotation(cam, cam_rot)

        if others:
            info['Coverage_rate'] = others['Coverage_rate']
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']
            info['Camera_local_goal'] = others['Camera_local_goal']
            info['cost'] = others['cost']

        info['Reward'] = np.array(r)
        info['Global_reward'] = np.array(gr)

        gt_directions = []
        gt_distance = []
        for i, cam in enumerate(self.cam_id):
            # for target navigation
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(self.get_location(cam) + self.get_rotation(cam))

        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Distance'] = np.array(gt_distance)
        info['Directions'] = np.array(gt_directions)

        self.count_steps += 1
        # set your done condition
        if self.count_steps > self.max_steps:
            info['Done'] = True

        reward = info['Global_reward']

        state, self.state_dim = self.preprocess_pose(info, GoalMap=self.goals4cam)
        return state, reward, info['Done'], info

    def get_baseline_action(self, cam_loc_rot, goals, i):
        camera_target_visible = []
        for k, v in self.Camera_target_dict.items():
            camera_target_visible += v

        goal_ids = goal_id_filter(goals)
        if len(goal_ids) != 0:
            if self.slave_rule:
                target_position = (self.target_pos_list[goal_ids]).mean(axis=0)  # avg pos: [x,y,z]

                angle_h = self.get_hori_direction(cam_loc_rot, target_position)

                action_h = angle_h // self.rotation_scale
                action_h = np.clip(action_h, -1, 1)
                action_h *= self.rotation_scale
                action = [action_h]
            else:
                tmp = []
                for j in range(len(self.target_pos_list[goal_ids])):
                    tar_p = self.target_pos_list[goal_ids][j]
                    angle_h = self.get_hori_direction(cam_loc_rot, tar_p)
                    d = self.get_distance(cam_loc_rot, tar_p)
                    tmp.append([i / 4, j / 5, angle_h / 180, d / 2000])
                target = np.zeros((1, self.num_target, 4))
                target[0, :len(tmp)] = tmp
                values, actions, entropies, log_probs = self.slave(torch.from_numpy(target).float().to(self.device),
                                                                   test=True)
                action = actions.item()
                action = np.array(self.discrete_actions[action]) * self.rotation_scale
        else:
            action = np.array(
                self.discrete_actions[np.random.choice(range(len(self.discrete_actions)))]) * self.rotation_scale

        return action

    def simulate(self, GoalMap, cam_info, keep=-1):
        cost = 0
        gre = np.array([0.0])
        for _ in range(keep):
            # camera move
            Cam_Pose = []
            for i, cam in enumerate(self.cam_id):
                cam_loc, cam_rot = cam_info[i]
                action = self.get_baseline_action(cam_loc + cam_rot, GoalMap[i], i)
                if action[0] != 0:
                    cost += 1
                cam_rot[0] += action[0]
                cam_info[i] = cam_loc, cam_rot
                Cam_Pose.append(cam_loc + cam_rot)

            # target move
            self.target_move()
            #
            r, gr, others = self.multi_reward(cam_info, GoalMap)
            gre += gr

            # render
            if self.render:
                render(Cam_Pose, np.array(self.target_pos_list), gr, self.goals4cam, save=self.render_save)

        cost = cost / keep
        others['cost'] = cost

        return r, gre / keep, others, cam_info

    def close(self):
        pass

    def seed(self, para):
        pass

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def angle_reward(self, angle_h, d):
        hori_reward = 1 - abs(angle_h) / 45.0
        visible = hori_reward > 0 and d <= self.visual_distance
        if visible:
            reward = np.clip(hori_reward, -1, 1)  # * (self.visual_distance-d)
        else:
            reward = -1
        return reward, visible

    def multi_reward(self, cam_info, goals4cam):
        camera_local_rewards = []
        camera_local_goal = []

        camera_target_dict = {}
        target_camera_dict = {}
        captured_targets = []
        coverage_rate = []
        for i, cam in enumerate(self.cam_id):
            cam_loc, cam_rot = cam_info[i]
            camera_target_dict[i] = []
            local_rewards = []
            captured_num = 0
            goal_num = 0
            for j in range(self.num_target):
                if not target_camera_dict.get(j):
                    target_camera_dict[j] = []
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                reward, visible = self.angle_reward(angle_h, d)
                if visible:
                    camera_target_dict[i].append(j)
                    target_camera_dict[j].append(i)
                    coverage_rate.append(j)
                    if goals4cam is None or goals4cam[i][j] > 0:
                        captured_targets.append(j)
                        captured_num += 1

                if goals4cam is None and visible or goals4cam is not None and goals4cam[i][j] > 0:
                    local_rewards.append(reward)
                    goal_num += 1
            camera_local_goal.append(captured_num / goal_num if goal_num != 0 else -1)
            camera_local_rewards.append(np.mean(local_rewards) if len(local_rewards) > 0 else 0)
            camera_local = camera_local_rewards

        # real coverage rate
        coverage_rate = len(set(coverage_rate)) / self.num_target

        camera_global_reward = [coverage_rate]
        if len(set(captured_targets)) == 0:
            camera_global_reward = [-0.1]

        return camera_local, camera_global_reward, {'Camera_target_dict': camera_target_dict,
                                                    'Target_camera_dict': target_camera_dict,
                                                    'Coverage_rate': coverage_rate,
                                                    'Captured_targetsN': len(set(captured_targets)),
                                                    'Camera_local_goal': camera_local_goal
                                                    }

    def preprocess_pose(self, info, GoalMap=None):
        cam_pose_info = np.array(info['Cam_Pose'])
        target_pose_info = np.array(info['Target_Pose'])
        angles = info['Directions']
        distances = info['Distance']

        camera_num = len(cam_pose_info)
        target_num = len(target_pose_info)

        # normalize center
        center = np.mean(cam_pose_info[:, :2], axis=0)
        cam_pose_info[:, :2] -= center
        if target_pose_info is not None:
            target_pose_info[:, :2] -= center

        # scale
        norm_d = int(max(np.linalg.norm(cam_pose_info[:, :2], axis=1, ord=2))) + 1e-8
        cam_pose_info[:, :2] /= norm_d
        if target_pose_info is not None:
            target_pose_info[:, :2] /= norm_d

        state_dim = 4
        feature_dim = target_num * state_dim
        state = np.zeros((camera_num, feature_dim))
        for cam_i in range(camera_num):
            # target info
            target_info = []
            for target_j in range(target_num):
                [angle_h] = angles[cam_i, target_j]
                target_angle = [cam_i / camera_num, target_j / target_num, angle_h / 180]
                line = target_angle + [distances[cam_i, target_j] / 2000]  # 2000 is related with the area of cameras
                target_info += line
            target_info = target_info + [0] * (feature_dim - len(target_info))
            state[cam_i] = target_info
        state = state.reshape((camera_num, target_num, state_dim))
        return state, state_dim


class GoalNavAgent(object):

    def __init__(self, id, action_space, goal_area, goal_list=None):
        self.id = id
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.goal_list = goal_list
        self.goal = self.generate_goal(self.goal_area)

        self.max_len = 100

    def act(self, pose):
        self.step_counter += 1
        if len(self.pose_last[0]) == 0:
            self.pose_last[0] = np.array(pose)
            self.pose_last[1] = np.array(pose)
            d_moved = 30
        else:
            d_moved = min(np.linalg.norm(np.array(self.pose_last[0]) - np.array(pose)),
                          np.linalg.norm(np.array(self.pose_last[1]) - np.array(pose)))
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(pose)
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area)
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)

            self.step_counter = 0

        delt_unit = (self.goal[:2] - pose[:2]) / np.linalg.norm(self.goal[:2] - pose[:2])
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [velocity, delt_unit[0], delt_unit[1]]

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]

    def generate_goal(self, goal_area):
        if self.goal_list and len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            x = np.random.randint(goal_area[0], goal_area[1])
            y = np.random.randint(goal_area[2], goal_area[3])
            goal = np.array([x, y])
        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 5
