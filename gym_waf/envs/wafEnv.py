#-*- coding:utf-8 –*-
import numpy
import random

import gym
from gym import spaces
from gym_waf.envs.features import Features
from gym_waf.envs.waf import Waf_Check
from gym_waf.envs.xss_manipulator import Xss_Manipulator

from sklearn.model_selection import train_test_split

# 由本地文件构建样本集
samples_file = "xss-samples-all.txt"    # 样本文件名
samples = []    # 样本集，开始为空
# 打开样本文件，逐行添加进样本集
with open(samples_file) as f:
    for line in f:
        line = line.strip('\n')
        print("添加xss样本：" + line)
        samples.append(line)

# 划分训练集和测试集，测试集占总体40%
samples_train, samples_test = train_test_split(samples, test_size=0.4)

# 构造动作速查表
ACTION_LOOKUP = {i:act for i,act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())}    # key为原动作字典的下标0123，value为原动作字典的key即免杀操作名

class WafEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }
    def __init__(self):
        # 构造动作空间
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))     # 离散类型、大小为自定义的免杀操作的数目
        
        self.current_sample = ""     # 当前正处理的样本
        self.features_extra = Features()    # 实例化特征提取的对象
        self.waf_checker = Waf_Check()
        # 根据动作修改当前样本免杀
        self.xss_manipulatorer = Xss_Manipulator()
        self._seed = 2
        # 初始化环境
        self.reset()    # 返回的是观测值空间(字符串的特征向量)

    def step(self, action):

        r = 0       # 默认奖励为0，免杀成功设置为10
        is_gameover = False     # 默认本轮学习未结束

        _action = ACTION_LOOKUP[action]     # 通过转换表得到免杀操作名，由数字到名称
        self.current_sample = self.xss_manipulatorer.modify(self.current_sample, _action)   # 将样本和动作发往免杀模块，依据动作对样本做相应修改
        if not self.waf_checker.check_xss(self.current_sample):     # 检测是否免杀绕过成功
            r = 10      # 给奖励
            is_gameover = True      # 免杀成功，则本轮结束
            print("很好，成功绕过waf%s" % self.current_sample)

        self.observation_space = self.features_extra.extract(self.current_sample)   # 得到新的观测值

        return self.observation_space, r, is_gameover, {}   # 返回新观测值、奖励、是否结束的标记及其他信息（元组表示，此处为空）

    def reset(self):
        # 选择当前样本并打印
        self.current_sample = random.choice(samples_train)  # 从训练集中随机选取一条样本作为当前处理样本
        print("当前处理样本为：" + self.current_sample)
        # 构造观测值空间（用特征向量）
        self.observation_space = self.features_extra.extract(self.current_sample)   # shape为(1,257)；第一维表字符串长度，其余256表字符出现次数的平均值

        return self.observation_space

    def render(self, mode='human', close=False):
        return