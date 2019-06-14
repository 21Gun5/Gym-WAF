from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, ELU, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard

def generate_dense_model(input_shape, layers, nb_actions):  # shape：输入的特征向量的维度，为(1,1,257)；layers：神经网络层数；nb_action：动作空间大小，即动作数
    model = Sequential()    # 采用顺序模型
    model.add(Flatten(input_shape=input_shape))     # 将输入展平，即多维将为一维，(1,1,257)至257
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

model = generate_dense_model((1,1,257), [5], 4)
model.compile(loss='mse', optimizer='sgd')


tbCallBack = TensorBoard(log_dir='./log', write_images=1, histogram_freq=1)