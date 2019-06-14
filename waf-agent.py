#-*- coding:utf-8 –*-
import gym
import time
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, ELU, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import gym_waf.envs.wafEnv
from gym_waf.envs.wafEnv  import samples_test, samples_train
from gym_waf.envs.features import Features
from gym_waf.envs.waf import Waf_Check
from gym_waf.envs.xss_manipulator import Xss_Manipulator

ENV_NAME = 'Waf-v0' # 之前注册的环境名，要按这样的形式命名，否则报错

# 尝试的最大次数
nb_max_episode_steps_train = 3     # fit训练时用到，在一次学习周期中的最大步数(默认一直学习直到“死”)
nb_max_episode_steps_test = 3

# 构造动作速查表
ACTION_LOOKUP = {i:act for i,act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())} # key为原动作字典的下标0123，value为原动作字典的key即免杀操作名

def generate_dense_model(input_shape, layers, nb_actions):  # shape：输入的特征向量的维度，为(1,1,257)；layers：神经网络层数；nb_action：动作空间大小，即动作数
    model = Sequential()    # 采用顺序模型
    model.add(Flatten(input_shape=input_shape))     # 将输入展平，即多维将为一维，(1,1,257)-257
    model.add(Dropout(0.1))     # 防止过拟合

    for layer in layers:
        print(layer)
        model.add(Dense(layer))
        model.add(BatchNormalization())
        model.add(ELU(alpha=1.0))

    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model

def train_dqn_model(layers, rounds=10000):
    env = gym.make(ENV_NAME)    # 创建环境
    env.seed(1)
    nb_actions = env.action_space.n     # 动作空间中包含几个动作，即有几种免杀操作
    window_length = 1       # 窗口长度，后面创建记忆体时用，通常设置为1

    # 打印动作、观测值相关信息
    print("免杀操作的个数：")
    print(nb_actions)   # 免杀操作的个数，可自行增加
    print("观测值空间形状：")
    print(env.observation_space.shape)      # 为(1,257)

    # 创建神经网络模型
    model = generate_dense_model((window_length,)+env.observation_space.shape, layers, nb_actions)
    # 指定选择策略(用于选择动作)
    policy = EpsGreedyQPolicy()     # 指定为∈贪婪算法，默认为贪婪算法
    # 创建记忆体（记忆即经验，包括当前状态、动作、下一个状态和奖励等信息）
    memory = SequentialMemory(limit=256, window_length=window_length)  # 记忆体大小为256，超过时，按照先进先出的原则丢弃；窗口长度通常设置为1；
    # 实例化agent智能体
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=16,
                     enable_double_dqn=True, enable_dueling_network=True, dueling_type='avg',
                     target_model_update=1e-2, policy=policy, batch_size=16)
    # 编译神经网络
    agent.compile(RMSprop(lr=1e-3), metrics=['mae'])    # 指定优化器为RMSprop且学习率为0.001、模型评估标准为mae平均绝对误差

    # 可视化神经网络结构：fit时加参数callbacks=[tb_cb]、tensorboard --logdir ./tmp/log、6006端口访问(待学习)
    # tb_cb = TensorBoard(log_dir='/tmp/logs', write_images=1, histogram_freq=1)

    # 强化学习的训练（keras-rl中训练与测试中严格分开，测试用test）
    # 环境、训练步数（而非学习次数)、一个学习周期内最多多少步、是否可视化、调试信息详细程度(0为不显示，2为全部显示)
    agent.fit(env, nb_steps=rounds, nb_max_episode_steps=nb_max_episode_steps_train, visualize=False, verbose=2) 

    # 强化学习的测试（keras-rl中训练与测试中严格分开，测试用test）
    # agent.test(env, nb_episodes=100)  # 测试的次数

    # test_samples = samples_test
    features_extra = Features()     # 特征向量
    waf_checker = Waf_Check()   # waf检验免杀效果
    xss_manipulatorer = Xss_Manipulator()   # 根据动作修改当前样本，来达到免杀

    success = 0     # 免杀成功数
    sum = 0     # 总数目

    shp = (1,) + tuple(model.input_shape[1:])

    for sample in samples_test:
        sum += 1

        for _ in range(nb_max_episode_steps_test):
            if not waf_checker.check_xss(sample) :
                success += 1
                print(sample)
                break

            f = features_extra.extract(sample).reshape(shp)
            act_values = model.predict(f)
            action = np.argmax(act_values[0])
            sample = xss_manipulatorer.modify(sample,ACTION_LOOKUP[action])

    print("总数量：{} 成功：{}".format(sum,success))

    return agent, model


if __name__ == '__main__':
    agent1, model1 = train_dqn_model([5], rounds=100)
    model1.save('waf-v0.h5', overwrite=True)





