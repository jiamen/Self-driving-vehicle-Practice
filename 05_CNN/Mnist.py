
from __future__ import print_function

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD            # 随机梯度下降函数（Stochastic Gradient Descent，SGD）
from matplotlib import pyplot as plt


# 三层网络的小改动——深度前馈网络
from keras.layers import Dropout
from keras.optimizers import RMSprop


batch_size = 128        # 训练批次大小
num_classes = 10        # 待分类数量
epochs = 20             # 训练次数


# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_train, y_test) = mnist.load_data()
f = np.load("mnist.npz")
x_train, y_train = f['x_train'], f['y_train']
x_test,  y_test  = f['x_test'], f['y_test']

print(x_train.shape, x_test.shape)  # (60000, 28, 28)   (10000, 28, 28)   数据是28×28的图像
print(y_train.shape, y_test.shape)  # (60000,)          (10000,)          标签

"""
display 16 samples and labels.
"""
def show_sample(samples, labels):
    plt.figure(figsize=(12, 12))
    for i in range(len(samples)):
        plt.subplot(4, 4, i+1)                  # 指定显示位置
        plt.imshow(samples[i], cmap='gray')     # 以 灰度图形式 显示 数据
        plt.title(labels[i])                    # 显示标题
    plt.show()


show_sample(x_train[:16], y_train[:16])


x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# 将样本归一化
x_train /= 255
x_test  /= 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape, x_test.shape)      # (60000, 784)   (10000, 784)
print(y_train.shape, y_test.shape)      # (60000, 10)    (10000, 10)


model = Sequential()
model.add(Dense(15, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

### print the keys contained in the history object
print(history.history.keys())


def plot_training(history):
    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


plot_training(history=history)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy', score[1])


# ==================================================================================================
# 改进上述模型，得到深度前馈神经网络：
# 1. 将网络层数加深，一个隐含层变成两个隐含层；
# 2. 将每一个隐含层的神经元数量扩大到512个；
# 3. 在每一个隐含层跟后面使用一种叫作Dropout的正则化技术；
# 4. 使用一种SGD变体  —— ——  RMSprop

model_improve = Sequential()
model_improve.add(Dense(512, activation='relu', input_shape=(784,)))
model_improve.add(Dropout(0.2))
model_improve.add(Dense(512, activation='relu'))
model_improve.add(Dropout(0.2))
model_improve.add(Dense(num_classes, activation='softmax'))


model_improve.summary()

model_improve.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history_improve = model_improve.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_test, y_test))


### print the keys contained in the history object
print(history_improve.history.keys())
plot_training(history=history_improve)
model_improve.save('model.json')


score = model_improve.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



result = model_improve.predict(x_test[:16])
result = np.argmax(result, 1)
print('predict: ', result)
true = np.argmax(y_test[:16], 1)
print('true: ', true)


