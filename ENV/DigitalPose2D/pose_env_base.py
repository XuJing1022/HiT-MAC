import os
import json
import math
import pickle
import random
import time

import torch

import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

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
                 slave_rule=False
                 ):

        self.nav = nav
        self.reset_type = reset_type
        self.ENV_PATH = 'ENV/DigitalPose2D'

        if setting_path:
            self.SETTING_PATH = setting_path  # "C:\\Users\\v-jixu7\\codes\\master4goal\\ENV\\DigitalPose\\PoseEnvLarge_multi.json"
        else:
            self.SETTING_PATH = os.path.join(self.ENV_PATH, config_path)
        with open(self.SETTING_PATH, encoding='utf-8') as f:  # /home/jill/Desktop/Projects/a3c_continuous/
            setting = json.load(f)

        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.discrete_actions_target = setting['discrete_actions_target']
        self.continous_actions = setting['continous_actions']
        self.continous_actions_player = setting['continous_actions_player']
        self.max_steps = setting['max_steps']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.height = setting['height']
        # self.pitch = setting['pitch']
        self.objects_env = setting['objects_list']
        self.reset_area = setting['reset_area']
        self.num_target = len(self.target_list)
        self.n = len(self.cam_id)
        self.cam_height = [setting['height'] for i in range(self.n)]

        # index = np.random.choice(range(len(setting['cam_area'])), self.num_cam, replace=False)
        self.cam_area = np.array(setting['cam_area'])

        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.exp_distance = setting['exp_distance']
        self.visual_distance = setting['visual_distance']
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], 1000)

        # define action space
        self.action_space = [spaces.Discrete(2) for i in range(self.n * self.num_target)]  # Box(low=-1.0, high=1.0, shape=(self.num_cam, self.num_target), dtype=np.float32)
        self.target_action_space = [spaces.Discrete(len(self.discrete_actions_target)) for i in range(self.num_target)]
        self.rotation_scale = setting['rotation_scale']

        # define observation space
        self.state_dim = 4
        self.observation_space = np.zeros((self.n, self.num_target, self.state_dim), int)

        self.action_type = setting['action_type']
        self.render_save = render_save

        self.cam = dict()
        for i in range(len(self.cam_id)+1):
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
                'logs/Pose-v0/Oct20_23-26/best.pth',
                map_location=lambda storage, loc: storage)
            self.slave.load_state_dict(saved_state['model'], strict=True)
            self.slave.eval()

    def set_location(self, cam_id, loc):
        self.cam[cam_id]['location'] = loc

    def get_location(self, cam_id):
        return self.cam[cam_id]['location']

    def set_rotation(self, cam_id, rot):
        for i in range(len(rot)):
            if rot[i]>180:
                rot[i] -= 360
            if rot[i]<-180:
                rot[i] += 360
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation']

    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt)/np.pi*180-current_pose[2]
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

        # z = np.random.randint(self.cam_area[0][4], self.cam_area[0][5])
        for i, cam in enumerate(self.cam_id):
            cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                       np.random.randint(self.cam_area[i][2], self.cam_area[i][3])
                       ]
            self.set_location(camera_id_list[i], cam_loc)  # shuffle

        for i, cam in enumerate(self.cam_id):
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)  # TODO DEBUG

            # angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[i])
            # start with non-focusing
            angle_h = np.random.randint(-180, 180)
            cam_rot[0] += angle_h*1.0

            self.set_rotation(cam, cam_rot)

        self.count_steps = 0
        self.early_stop = 20
        self.goal_keep = 0
        self.goals4cam = np.ones([self.n, self.num_target])

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_reward=[0 for i in range(self.num_target)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
            Depth=[],
            Gates=np.zeros((self.n, 1), float),
            Focus=[],
            Next_Actions=[0 for i in range(self.n)]
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

        # Target_mutual_distance
        gt_target_mu_distance = np.zeros([self.num_target, self.num_target])
        for i in range(self.num_target):
            for j in range(i+1):
                d = self.get_distance(self.target_pos_list[i], self.target_pos_list[j])
                gt_target_mu_distance[i, j] = d
                gt_target_mu_distance[j, i] = d
        info['Target_mutual_distance'] = gt_target_mu_distance

        info['Directions'] = np.array(gt_directions)
        info['Distance'] = np.array(gt_distance)
        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Reward'], info['Target_reward'], info['Global_reward'], others = self.multi_reward(cam_info, self.goals4cam)
        if others:
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']

        state, self.state_dim = self.preprocess_pose(info)
        return state

    # def goal_id_filter(self, goals):
    #     return np.where(goals > 0.5)[0]

    def format_goalmap(self, subset, row, column):
        Gmap = np.zeros((row, column))
        for i in subset:
            r = i//column
            c = i%column
            Gmap[r,c] = 1  # positive value

        return Gmap

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
                action = self.random_agents[i].act(loc)  # TODO test Goal & Random [v, h_delta, p_delta, v_delta]

                target_hpr_now = np.array(action[1:])
                delta_x = target_hpr_now[0] * action[0] * delta_time
                delta_y = target_hpr_now[1] * action[0] * delta_time
                while loc[0] + delta_x < self.reset_area[0] or loc[0] + delta_x > self.reset_area[1] or \
                        loc[1] + delta_y < self.reset_area[2] or loc[1] + delta_y > self.reset_area[3]:
                    # print('retry: ', loc, action)
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
            Target_reward=[0 for i in range(self.num_target)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps,
            Depth=[],
            Gates=np.zeros((self.n, 1), float),
            Focus=[],
            Next_Actions=[0 for i in range(self.n)]
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
        r, tr, gr, others, cam_info = self.simulate(self.goals4cam, cam_info, keep=10)
        for i, cam in enumerate(self.cam_id):
            cam_loc, cam_rot = cam_info[i]
            self.set_rotation(cam, cam_rot)

        if others:
            info['Coverage_rate'] = others['Coverage_rate']
            info['Avg_accuracy'] = others['Avg_accuracy']
            info['Camera_target_dict'] = self.Camera_target_dict = others['Camera_target_dict']
            info['Target_camera_dict'] = self.Target_camera_dict = others['Target_camera_dict']
            info['Camera_local_goal'] = others['Camera_local_goal']
            info['cost'] = others['cost']

        info['Reward'] = np.array(r)
        info['Target_reward'] = np.array(tr)
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

            info['Cam_Pose'].append(self.get_location(cam)+self.get_rotation(cam))

        info['Target_Pose'] = np.array(self.target_pos_list)  # copy.deepcopy
        info['Distance'] = np.array(gt_distance)
        info['Directions'] = np.array(gt_directions)

        # Target_mutual_distance
        gt_target_mu_distance = np.zeros([self.num_target, self.num_target])
        for i in range(self.num_target):
            for j in range(i+1):
                d = self.get_distance(self.target_pos_list[i], self.target_pos_list[j])
                gt_target_mu_distance[i, j] = d
                gt_target_mu_distance[j, i] = d
        info['Target_mutual_distance'] = gt_target_mu_distance

        self.count_steps += 1
        if max(info['Global_reward'].flatten()) < -1 or not info['Reward'].all(0) > 0:
            self.early_stop -= 1
        else:
            self.early_stop = 10

        # set your done condition
        if self.count_steps > self.max_steps: # or self.early_stop == 0:
            info['Done'] = True

        reward = info['Global_reward']

        # # show
        # render(info['Cam_Pose'], info['Target_Pose'], reward, self.goals4cam, savestep=savestep)

        state, self.state_dim = self.preprocess_pose(info, GoalMap=self.goals4cam)
        return state, reward, info['Done'], info  # TODO: train single with local reward or global reward?

    def get_baseline_action(self, cam_loc_rot, goals, i, optimal=False):
        # move camera according to the target_pos_list and visible_map

        camera_target_visible = []
        for k, v in self.Camera_target_dict.items():
            camera_target_visible += v

        goal_ids = goal_id_filter(goals)  # list(set(np.where(goals > 0.5)[0]))  # & set(camera_target_visible)
        # sum = goals[goals > 0].sum()
        if len(goal_ids) != 0:
            if self.slave_rule:
                target_position = (self.target_pos_list[goal_ids]).mean(axis=0)  # avg pos: [x,y,z]

                angle_h = self.get_hori_direction(cam_loc_rot, target_position)

                action_h = angle_h // self.rotation_scale
                if optimal is False:
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
                target = np.zeros((1,self.num_target, 4))
                target[0,:len(tmp)] = tmp
                values, actions, entropies, log_probs = self.slave(torch.from_numpy(target).float().to(self.device), test=True)
                action_h_pred = actions.item()
                # if action_h_pred != action_h:
                #     print(action_h, action_h_pred)
                # action_h = action_h_pred
                # action_h *= self.rotation_scale
                # action = [action_h]
                action = np.array(self.discrete_actions[action_h_pred])*self.rotation_scale
        else:
            action = np.array(self.discrete_actions[np.random.choice(range(len(self.discrete_actions)))])*self.rotation_scale

        return action

    def simulate(self, GoalMap, cam_info, keep=-1):
        cost = 0
        gre = np.array([0.0])
        for _ in range(keep):
            # camera move
            Cam_Pose = []
            for i, cam in enumerate(self.cam_id):
                cam_loc, cam_rot = cam_info[i]
                action = self.get_baseline_action(cam_loc + cam_rot, GoalMap[i], i, optimal=keep == -1)
                if action[0] != 0:
                    cost += -0.1
                cam_rot[0] += action[0]
                cam_info[i] = cam_loc, cam_rot
                Cam_Pose.append(cam_loc+cam_rot)

            # target move
            self.target_move()
            #
            r, tr, gr, others = self.multi_reward(cam_info, GoalMap)  # , cost)
            gre+=gr

            # render
            render(Cam_Pose, np.array(self.target_pos_list), gr, self.goals4cam, save=True)

        cost = cost/keep

        # r: every camera complete its goal; [camera_num]
        # tr: target reward [max(r)]; [target_num]
        # gr: coverage rate; [1]
        # r, tr, gr, others = self.multi_reward(cam_info, GoalMap)  # , cost)
        others['cost'] = cost

        return r, tr, gre/keep, others, cam_info

    def render(self, camera_pos, target_pos, Target_camera_dict=None, Camera_target_dict=None, distance=None, reward=None, goal=None, savestep=None):
        camera_pos = np.array(camera_pos)
        target_pos = np.array(target_pos)

        camera_pos[:, :2] /= 1000.0
        target_pos[:, :2] /= 1000.0

        length = 600
        area_length = 1  # for random cam loc
        target_pos[:, :2] = (target_pos[:, :2] + 1) / 2
        camera_pos[:, :2] = (camera_pos[:, :2] + 1) / 2

        img = np.zeros((length + 1, length + 1, 3)) + 255
        num_cam = len(camera_pos)
        num_target = len(target_pos)
        camera_position = [camera_pos[i][:2] for i in range(num_cam)]
        target_position = [target_pos[i][:2] for i in range(num_target)]

        camera_position = length * (1 - np.array(camera_position) / area_length) / 2
        target_position = length * (1 - np.array(target_position) / area_length) / 2
        abs_angles = [camera_pos[i][2] * -1 for i in range(num_cam)]

        color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                      'darkred': [128, 0, 0], 'yellow': [255, 255, 0], 'deeppink': [255, 20, 147]}

        angle = 90

        plt.cla()
        plt.imshow(img.astype(np.uint8))

        # get camera's view space positions
        visua_len = 100  # length of arrow
        L = 140  # length of arrow
        for i in range(num_cam):
            theta = abs_angles[i]  # -90
            theta -= 90
            dx = L * math.sin(theta * math.pi / 180)
            dy = L * math.cos(theta * math.pi / 180)

            the1 = theta-45
            dxy1 = [L * math.sin(the1 * math.pi / 180), L * math.cos(the1 * math.pi / 180)]

            the2 = theta+45
            dxy2 = [L * math.sin(the2 * math.pi / 180), L * math.cos(the2 * math.pi / 180)]

            color = 'b'
            if Camera_target_dict is not None and len(Camera_target_dict[i]) == 0:
                color = 'r'
            # plt.arrow(camera_position[i][0]+visua_len, camera_position[i][1]+visua_len, dx, dy, width=0.1, head_width=8, head_length=8,
            #           length_includes_head=True, color=color)
            plt.arrow(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len, dxy1[0], dxy1[1], width=0.1, head_width=8, head_length=8,
                      length_includes_head=True, color=color)
            plt.arrow(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len, dxy2[0], dxy2[1], width=0.1, head_width=8, head_length=8,
                      length_includes_head=True, color=color)

            plt.annotate(str(i + 1), xy=(camera_position[i][0]+visua_len, camera_position[i][1]+visua_len),
                         xytext=(camera_position[i][0]+visua_len, camera_position[i][1]+visua_len), fontsize=10, color='blue')
            # plt.plot(camera_position[i][0], camera_position[i][1],'ro', color='blue')

        # plot reward
        if reward is not None:
            plt.text(10, 20, 'Reward: ' + str(reward), weight="bold", color="b")
        # plot target
        target_color_dict = {}
        for i in range(num_target):
            target_color_dict[i] = 'r'
        if distance is not None:
            for i in range(num_cam):
                for j in range(num_target):
                    if distance[i][j]<self.visual_distance:
                        target_color_dict[j] = (1.0, 215/255.0, 0.0)
        if Target_camera_dict is not None:
            for i in range(num_target):
                if len(Target_camera_dict[i]) > 0:
                    target_color_dict[i] = (0.0, len(Target_camera_dict[i]) / num_cam, 0.0)
        for i in range(num_target):
            plt.plot(target_position[i][0]+visua_len, target_position[i][1]+visua_len, color=target_color_dict[i], marker="o")
            plt.annotate(str(i + 1), xy=(target_position[i][0]+visua_len, target_position[i][1]+visua_len),
                         xytext=(target_position[i][0]+visua_len, target_position[i][1]+visua_len), fontsize=10, color='black')

        if goal is not None:
            for i in range(len(goal)):
                tmp = np.zeros(len(goal[i]))
                tmp[goal[i]>0.5] = 1
                plt.text(400, 500+i*30, str(tmp))

        if savestep:
            file_path = 'logs/img'
            file_name = '{}_{}.jpg'.format(datetime.now().strftime('%b%d_%H-%M-%S'), savestep)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.savefig(os.path.join(file_path, file_name))
        plt.pause(0.01)

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
            reward = np.clip(hori_reward, -1, 1)  #  * (self.visual_distance-d)
        else:  # 在视野范围外
            reward = -1
        return reward, visible

    def multi_reward(self, cam_info, goals4cam, cost=0):
        """
        track multiple targets, use 1) how many targets are captured by cameras;
        2) for every target, how many cameras are capturing; as rewards
        :return:
        """
        # generate reward
        camera_local_rewards = []
        camera_local_goal = []

        camera_target_dict = {}
        target_camera_dict = {}
        captured_targets = []
        camera_target_reward = []
        coverage_rate = []
        for i, cam in enumerate(self.cam_id):
            cam_loc, cam_rot = cam_info[i]
            camera_target_dict[i] = []
            local_rewards = []
            camera_target_reward.append([])
            captured_num = 0
            goal_num = 0
            for j in range(self.num_target):
                if not target_camera_dict.get(j):
                    target_camera_dict[j] = []
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                reward, visible = self.angle_reward(angle_h, d)
                if visible:  # 在视野范围内
                    camera_target_dict[i].append(j)
                    target_camera_dict[j].append(i)
                    coverage_rate.append(j)
                    if goals4cam is None or goals4cam[i][j] > 0:  # 如果非层级结构，则没有goal4cam
                        captured_targets.append(j)  # 训练时追求goal的完成度，即goal下的覆盖率
                        captured_num += 1

                if goals4cam is None and visible or goals4cam is not None and goals4cam[i][j] > 0:
                    local_rewards.append(reward)  # 每个相机的goal的完成度，即想追的是否都已进入画面；而非goal的部分不作排除要求，因为可能是target的路线自身造成的出现在画面中
                    goal_num += 1
                camera_target_reward[i].append(reward)
            camera_local_goal.append(captured_num/goal_num if goal_num != 0 else -1)  # 谁都不追的话，则最差；如果选的越多，此处能到满分越难
            camera_local_rewards.append(np.mean(local_rewards) if len(local_rewards)>0 else 0)
            camera_local = camera_local_rewards

        # real coverage rate
        coverage_rate = len(set(coverage_rate))/self.num_target
        # coverage filtered by goal map
        # coverage_rate = len(set(captured_targets))/self.num_target

        camera_global_reward = [coverage_rate+cost]  # 1)reward: [-1, 1], coverage
        if len(set(captured_targets)) == 0:
            camera_global_reward = [-0.1]

        # HWH: max(R_local) as target_local_rewards
        camera_target_reward = np.array(camera_target_reward)
        target_local_rewards = np.max(camera_target_reward, axis=0)

        # # contribution reward
        # for j in range(self.num_target):
        #     for i in target_camera_dict[j]:
        #         camera_global_reward[i] += 1/len(target_camera_dict[j])/self.num_target

        # # local error
        # for i in range(self.num_cam):
        #     if len(camera_local_rewards[i]) != 0:
        #         # print('camera_global_reward_{}'.format(i), np.mean(camera_local_rewards[i]))
        #         camera_global_reward[i] += (1/self.num_target) * np.mean(camera_local_rewards[i])
        #     else:
        #         camera_global_reward[i] += -0.1
        # print('after', camera_global_reward)

        return camera_local, target_local_rewards, camera_global_reward, {'Camera_target_dict': camera_target_dict,
                                                                          'Target_camera_dict': target_camera_dict,
                                                                          'Coverage_rate': coverage_rate,
                                                                          'Avg_accuracy': target_local_rewards.mean(),
                                                                          'Captured_targetsN': len(set(captured_targets)),
                                                                          'Camera_local_goal': camera_local_goal
                                                                          }

    def preprocess_pose(self, info, GoalMap=None):
        cam_pose_info = np.array(info['Cam_Pose'])
        target_pose_info = np.array(info['Target_Pose'])
        camera_target_dict = info.get('Camera_target_dict')
        target_camera_dict = info.get('Target_camera_dict')
        angles = info['Directions']
        distances = info['Distance']
        target_mutual_distance = info['Target_mutual_distance']

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

        target_captured = []
        for cam_i in range(camera_num):
            target_captured += camera_target_dict[cam_i]
        target_captured = list(set(target_captured))

        # keep the selected goal map
        if self.goal_keep > 0:
            self.goal_keep -= 1
            reset_target_selected = False
        else:
            self.goal_keep = self.KEEP
            reset_target_selected = True

        for cam_i in range(camera_num):
            # target_list = camera_target_dict[cam_i]
            target_list = np.argwhere(distances[cam_i]<=self.visual_distance).squeeze(-1)

            # # camera pose
            # camera_pose = []
            # for sub_i in range(cam_i, cam_i + camera_num):
            #     camera_pose += new_pose[sub_i % camera_num]  # 7
            #     break

            if GoalMap is None:
                if reset_target_selected:
                    target_isSelected_list = self.select_target_random(target_list, target_mutual_distance)
                    self.goals4cam[cam_i] = target_isSelected_list
                else:
                    target_isSelected_list = self.goals4cam[cam_i]
            else:
                target_isSelected_list = GoalMap[cam_i]
            # target info
            target_info = []
            # for target_j in target_list:
            for target_j in range(target_num):  # 所有相机知道所有target的位置
                if self.reset_type==0 and target_isSelected_list[target_j] == 0:
                    continue
                [angle_h] = angles[cam_i, target_j]
                target_angle = [cam_i / camera_num, target_j / target_num, angle_h / 180]  # , math.sin(angle_h), math.cos(angle_h), math.sin(angle_v), math.cos(angle_v)]
                label_current = 1 if cam_i in target_camera_dict[target_j] else -1
                label_global = 1 if cam_i in target_captured else -1
                line = target_angle + [distances[cam_i, target_j] / 2000]  #, label_current, label_global]  # 6
                target_info += line
            target_info = target_info + [0]*(feature_dim-len(target_info))
            state[cam_i] = target_info
        state = state.reshape((camera_num, target_num, state_dim))
        return state, state_dim

    def select_target_random(self, visible_target_list, target_mutual_distance):
        '''
        select target for camera to track. select one from visible_target_list, randomly sample others from target_mutual_distance
        :param target_list:
        :param target_mutual_distance:
        :return: 1: selected
        '''

        if len(visible_target_list) != 0:
            main_target = random.choice(visible_target_list)
        else:
            main_target = random.choice(range(len(target_mutual_distance)))

        # sample by probs
        probs = target_mutual_distance[main_target] / max(target_mutual_distance[main_target])
        target_selected = []
        for i in range(len(probs)):
            target_selected.append(np.random.choice([0, 1], p=[probs[i], 1-probs[i]]))

        # # sample one
        # target_selected = [0]*self.num_target
        # target_selected[main_target] = 1

        return target_selected


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.step_counter = 0
        self.keep_steps = 0
        self.action_space = action_space

    def act(self):
        self.step_counter += 1
        if self.step_counter > self.keep_steps:
            self.action = self.action_space.sample()
            self.keep_steps = np.random.randint(1, 10)
        return self.action

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0


class GoalNavAgent(object):  # TODO debug

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

        delt_unit = (self.goal[:2] - pose[:2])/np.linalg.norm(self.goal[:2] - pose[:2])
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [velocity, delt_unit[0], delt_unit[1]]

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[],[]]

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
