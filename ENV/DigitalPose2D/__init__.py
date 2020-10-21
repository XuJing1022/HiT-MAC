# -*- coding: UTF-8 -*-

"""
@version: python3
@author: Xu Jing
@contact: xujingdaily@gmail.com
@site: https://github.com/XuJing1022
@file: __init__.py.py
@created: 19-7-4 下午6:09
"""
from ENV.DigitalPose2D.pose_env_base import Pose_Env_Base


class Gym:
    def make(self, env_id, render_save):
        reset_type = env_id.split('-v')[1]
        env = Pose_Env_Base(int(reset_type),render_save=render_save)
        return env


gym = Gym()
