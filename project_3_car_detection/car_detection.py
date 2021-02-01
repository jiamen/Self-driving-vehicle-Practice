import sys
# print(sys.path)
# sys.path.append('/home/zlc/anaconda3/envs/tf1.8/lib/python3.6/site-packages')

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

import keras

print(keras.__version__)
# from keras import backend as K        # 测试有无可用的gpu
# print(K.tensorflow_backend._get_available_gpus())
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils.utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box

keras.backend.set_image_data_format('channels_first')
print(keras.backend.image_data_format())

model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))

print(model.summary())

# https://pan.baidu.com/s/1o9twnPo
load_weights(model, './yolo-tiny.weights')

imagePath = './test_images/test1.jpg'
image = plt.imread(imagePath)
image_crop = image[300:650, 500:, :]
resized = cv2.resize(image_crop, (448, 448))

batch = np.transpose(resized, (2, 0, 1))
batch = 2 * (batch / 255.) - 1
batch = np.expand_dims(batch, axis=0)
out = model.predict(batch)

boxes = yolo_net_out_to_car_boxes(out[0], threshold=0.17)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.imshow(image)
ax2.imshow(draw_box(boxes, plt.imread(imagePath), [[500, 1200], [300, 650]]))

plt.show()

images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
batch = np.array([np.transpose(cv2.resize(image[300:650, 500:, :], (448, 448)), (2, 0, 1))
                  for image in images])
batch = 2 * (batch / 255.) - 1
out = model.predict(batch)
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 10))
for i, ax in zip(range(len(batch)), [ax1, ax2, ax3, ax4, ax5, ax6]):
    boxes = yolo_net_out_to_car_boxes(out[i], threshold=0.17)
    ax.imshow(draw_box(boxes, images[i], [[500, 1280], [300, 650]]))

plt.show()


def frame_func(image):
    crop = image[:, 80:438, :]
    resized = cv2.resize(crop, (448, 448))
    batch = np.array([resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]])
    batch = 2 * (batch / 255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold=0.17)
    return draw_box(boxes, image, [[80, 438], [0, 288]])

# project_video_output = './project_video_output.mp4'
# clip1 = VideoFileClip("./project_video.mp4")
project_video_output = './cartail0.mp4'
clip1 = VideoFileClip("./cartail0.avi")


lane_clip = clip1.fl_image(frame_func)      # NOTE: this function expects color images!!
lane_clip.write_videofile(project_video_output, audio=False)

