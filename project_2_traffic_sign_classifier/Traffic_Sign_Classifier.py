# -*- coding: UTF-8 -*-
"""
    Project 2：目标识别之交通指示牌识别
    自动识别交通指示牌是实现完全自动驾驶的一个关键和必要的技术. 这个项目是一个典型的深度学习在计算机视觉领域的应用, 具有广泛的应用范围.
    通过此项目的学习, 您能够将掌握的知识用于人脸识别, 商品识别(无人超市), 鉴黄, 安全监控, OCR等许多领域.
        Part 1: 数据增强
        Part 2: 图像预处理
        Part 3: 设计卷积神经网络进行神经学习
        Part 4: Debug, 根据validation和实际使用的结果做调整
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
print(cv2.__version__)
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm
import os


""" 第0步：加载数据 """
def load_traffic_sign_data(training_file, testing_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test  = pickle.load(f)
    return train, test

# load pickled data  加载数据
train, test = load_traffic_sign_data('./traffic-signs-data/train.p', './traffic-signs-data/test.p')
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


""" 第1步： Dataset Summary & Exploration 数据集摘要与探索 
The pickled data is a dictionary with 4 key/value pairs:
    ①'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
    ②'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
    ③'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
    ④'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 
    THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES.
"""
# Number of examples
n_train, n_test = X_train.shape[0], X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many classes?
n_classes = np.unique(y_train).shape[0]

print("Number of training examples = ", n_train)        # 39209
print("Number of testing  examples = ", n_test)         # 12630
print("Image data shape = ", image_shape)               # (32, 32, 3)
print("Number of classes = ", n_classes)                # 43

# show a random sample from each class of the traffic sign dataset
rows, cols = 4, 12
fig, ax_array = plt.subplots(rows, cols)
# print(fig)          # Figure(640x480)
# print(ax_array)     # 4×12的数组
plt.suptitle('Random samples from set (one for each class)')

for class_idx, ax in enumerate(ax_array.ravel()):       # ravel使更复杂; 使更纷乱;  将多维数组转换成
    if class_idx < n_classes:
        # show a random image of the current class
        cur_X = X_train[y_train == class_idx]
        cur_img = cur_X[np.random.randint(len(cur_X))]
        ax.imshow(cur_img)
        ax.set_title('{:02d}'.format(class_idx))
    else:
        ax.axis('off')

# hide both x and y ticks(发出滴答声; 滴答地走时; 标记号; 打上钩; 打对号;)   隐藏每张照片的xy坐标值
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.show()


# bart-chart of classes distribution
train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)
for c in range(n_classes):
    train_distribution[c] = np.sum(y_train == c) / n_train
    test_distribution[c]  = np.sum(y_test == c) / n_test

fig, ax = plt.subplots()

col_width = 0.5
bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
bar_test  = ax.bar(np.arange(n_classes)+col_width, test_distribution, width=col_width, color='b')

ax.set_ylabel('Percentage of presence')
ax.set_xlabel('Class Label')
ax.set_title('Classes distribution in traffic-sign dataset')
ax.set_xticks(np.arange(0, n_classes, 5) + col_width)
ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
ax.legend( (bar_train[0], bar_test[0]), ('train set', 'test set') )
plt.show()


##  图像预处理  ##
def preprocess_features(X, equalize_hist=True):
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X], dtype="object")

    # adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X], dtype="object")
    X = np.float32(X)

    # standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X

X_train_norm = preprocess_features(X_train)
X_test_norm  = preprocess_features(X_test)

"""
Question 1
Describe how you preprocessed the data. Why did you choose that technique?
    Answer: Following this paper [Sermanet, LeCun] I employed three main steps of feature preprocessing:
        1) each image is converted from RGB to YUV color space, then only the Y channel is used. This choice can sound at first suprising, 
        but the cited paper shows how this choice leads to the best performing model. This is slightly counter-intuitive, 
        but if we think about it arguably we are able to distinguish all the traffic signs just by looking to the grayscale image.
        2) contrast of each image is adjusted by means of histogram equalization. This is to mitigate the numerous situation in which the image contrast is really poor.
        3) each image is centered on zero mean and divided for its standard deviation. 
        This feature scaling is known to have beneficial effects on the gradient descent performed by the optimizer.
"""

# split into train and validation 将数据集按照二八分，80%划分为训练集，20%划分为验证集
VAL_RATIO = 0.2
X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train, test_size=VAL_RATIO, random_state=0)

# create the generator to perform online data augmentation      进行数据增强操作
image_datagen = ImageDataGenerator(rotation_range=15.,
                                 zoom_range=0.2,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1)

# take a random image from the training set
img_rgb = X_train[0]

# plot the original image
plt.figure(figsize=(1, 1))
plt.imshow(img_rgb)
plt.title('Example of RGB image (class = {})'.format(y_train[0]))
plt.show()


# plot some randomly augmented images
rows, cols = 4, 10
fig, ax_array = plt.subplots(rows, cols)

for ax in ax_array.ravel():
    augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
    ax.imshow( np.uint8(np.squeeze(augmented_img)) )

plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.suptitle('Random example of data augmentation (starting from the previous image)')
plt.show()


"""
Question 2
    Describe how you set up the training, validation and testing data for your model. Optional: 
    If you generated additional data, how did you generate the data? Why did you generate the data? 
    What are the differences in the new dataset (with generated data) from the original dataset?
Answer: For the train and test split, I just used the ones provided, composed by 39209 and 12630 examples respectively.
    To get additional data, I leveraged on the ImageDataGenerator class provided in the Keras library. No need to re-invent the wheel! 
    In this way I could perform data augmentation online, during the training. 
    Training images are randomly rotated, zoomed and shifted but just in a narrow range, 
    in order to create some variety in the data while not completely twisting the original feature content. 
    The result of this process of augmentation is visible in the previous figure.
"""

# placeholders 定义3个占位符
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int32, shape=None)        # tf.placeholder 是 TensorFlow中的占位符，用于传入外部数据。
keep_prob = tf.placeholder(tf.float32)                  # dropout中保留的概率


def weight_variable(shape, mu=0, sigma=0.1):
    initialization = tf.truncated_normal(shape=shape, mean=mu, stddev=sigma)    # 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
    # 取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]。
    return tf.Variable(initialization)          # 创建变量，矩阵

def bias_variable(shape, start_val=0.1):
    initialization = tf.constant(start_val, shape=shape)
    return tf.Variable(initialization)

def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input=x, filter=W, strides=strides, padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# network architecture definition
def my_net(x, n_classes):
    ## 前两层是卷积 ##
    c1_out = 64
    conv1_W = weight_variable(shape=(3, 3, 1, c1_out))
    conv1_b = bias_variable(shape=(c1_out, ))
    conv1 = tf.nn.relu(conv2d(x, conv1_W) + conv1_b)

    pool1 = max_pool_2x2(conv1)
    drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    c2_out = 128
    conv2_W = weight_variable(shape=(3, 3, c1_out, c2_out))         # 卷积权重
    conv2_b = bias_variable(shape=(c2_out, ))                       # 偏差
    conv2   = tf.nn.relu( conv2d(drop1, conv2_W) + conv2_b )

    pool2 = max_pool_2x2(conv2)
    drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob)


    ## 后两层是全连接的 ##
    fc0 = tf.concat([flatten(drop1), flatten(drop2)], 1)

    fc1_out = 64
    fc1_W = weight_variable(shape=(fc0.shape[1].value, fc1_out))
    fc1_b = bias_variable(shape=(fc1_out, ))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    fc2_out = n_classes
    fc2_W = weight_variable(shape=(drop_fc1.shape[1].value, fc2_out))
    fc2_b = bias_variable(shape=(fc2_out, ))
    logits = tf.matmul(drop_fc1, fc2_W) + fc2_b

    return logits


# training pipeline
lr = 0.001
print('GPU', tf.test.is_gpu_available())        # 测试GPU是否可用
logits = my_net(x, n_classes=n_classes)         # 43
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)         # 通过交叉熵得到最终的标签

# 定义损失函数
loss_function = tf.reduce_mean(cross_entropy)
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)


# metrics and functions for model evaluation    模型评估的指标和功能
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = X_data.shape[0]
    total_accuracy = 0

    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCHSIZE):
        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += accuracy * len(batch_x)

    return total_accuracy / num_examples


# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()


# Training hyperparameters
BATCHSIZE = 128
EPOCHS = 30                 # 训练30次
BATCHES_PER_EPOCH = 5000    # 每次训练的最大批次是5000个

"""
# start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # △△△ 全局变量初始化 △△△

    for epoch in range(EPOCHS):
        print("EPOCH {} ...".format(epoch + 1))

        batch_counter = 0
        for batch_x, batch_y in tqdm(image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE)):
            batch_counter += 1
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            if batch_counter == BATCHES_PER_EPOCH:
                break

        # at epoch end, evaluate accuracy on both training and validation set
        # 每个批次训练后，评估训练和验证集的准确性
        train_accuracy = evaluate(X_train_norm, y_train)
        val_accuracy = evaluate(X_val_norm, y_val)
        print('Train Accuracy = {:.3f} - Validation Accuracy: {:.3f}'.format(train_accuracy, val_accuracy))

        # log current weights
        checkpointer.save(sess, save_path='./checkpoints/traffic_sign_model.ckpt', global_step=epoch)
"""

# testing the model
with tf.Session() as sess:
    # restore saved session with highest validation accuracy
    checkpointer.restore(sess, './checkpoints/traffic_sign_model.ckpt-27')

    test_accuracy = evaluate(X_test_norm, y_test)
    print('Performance on test set: {:.3f}'.format(test_accuracy))


new_images_dir = 'other_signs_for_test/'
# new_test_images = [os.path.join(new_images_dir, f) for f in os.listdir(new_images_dir)]
# new_test_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in new_test_images]
new_test_images = []
for image in os.listdir(new_images_dir):
    img = cv2.imread(new_images_dir + image)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_test_images.append(img)


# manually annotated labels for these new images
new_targets = [3, 27, 17, 13, 14]
# 60  三角人 红圈横杠 倒三角 stop
# 3    27     17      13   14
"""
Image 0 - Target = 03, Predicted = 25
Image 1 - Target = 27, Predicted = 28
Image 2 - Target = 17, Predicted = 12
Image 3 - Target = 13, Predicted = 13
Image 4 - Target = 14, Predicted = 14
"""

# plot new test images
fig, axarray = plt.subplots(1, len(new_test_images))

# plot new test images
for i, ax in enumerate(axarray.ravel()):
    ax.imshow(new_test_images[i])
    ax.set_title('{}'.format(i))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticks([]), ax.set_yticks([])

plt.show()



# first things first: feature preprocessing
new_test_images_norm = preprocess_features(new_test_images)

with tf.Session() as sess:

    # restore saved session
    checkpointer.restore(sess, './checkpoints/traffic_sign_model.ckpt-27')

    # predict on unseen images
    prediction = np.argmax(np.array(sess.run(logits, feed_dict={x: new_test_images_norm, keep_prob: 1.})), axis=1)

for i, pred in enumerate(prediction):
    print('Image {} - Target = {:02d}, Predicted = {:02d}'.format(i, new_targets[i], pred))

print('> Model accuracy: {:.02f}'.format(np.sum(new_targets==prediction)/len(new_targets)))




# visualizing softmax probabilities
with tf.Session() as sess:

    # restore saved session
    checkpointer.restore(sess, './checkpoints/traffic_sign_model.ckpt-27')

    # certainty of predictions
    K = 3
    top_3 = sess.run(tf.nn.top_k(logits, k=K), feed_dict={x: new_test_images_norm, keep_prob: 1.})

    # compute softmax probabilities
    softmax_probs = sess.run(tf.nn.softmax(logits), feed_dict={x: new_test_images_norm, keep_prob: 1.})

# plot softmax probs along with traffic sign examples
n_images = new_test_images_norm.shape[0]
fig, axarray = plt.subplots(n_images, 2)
plt.suptitle('Visualization of softmax probabilities for each example', fontweight='bold')
for r in range(0, n_images):
    axarray[r, 0].imshow(np.squeeze(new_test_images[r]))
    axarray[r, 0].set_xticks([]), axarray[r, 0].set_yticks([])
    plt.setp(axarray[r, 0].get_xticklabels(), visible=False)
    plt.setp(axarray[r, 0].get_yticklabels(), visible=False)
    axarray[r, 1].bar(np.arange(n_classes), softmax_probs[r])
    axarray[r, 1].set_ylim([0, 1])

plt.show()

# print top K predictions of the model for each example, along with confidence (softmax score)
for i in range(len(new_test_images)):
    print('Top {} model predictions for image {} (Target is {:02d})'.format(K, i, new_targets[i]))
    for k in range(K):
        top_c = top_3[1][i][k]
        print('   Prediction = {:02d} with confidence {:.2f}'.format(top_c, softmax_probs[i][top_c]))
