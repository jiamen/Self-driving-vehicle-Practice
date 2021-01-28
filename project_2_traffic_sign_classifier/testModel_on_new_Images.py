
### Load the images and plot them here
import os
import cv2
import matplotlib.pyplot as plt
from Traffic_Sign_Classifier import preprocess_features, my_net
import tensorflow as tf
import numpy as np


# load new images
new_images_dir = 'other_signs_for_test'
new_test_images = [os.path.join(new_images_dir, f) for f in os.listdir(new_images_dir)]
new_test_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in new_test_images]

# manually annotated labels for these new images
new_targets = [3, 27, 17, 13, 14]
# 60  三角人 红圈横杠 倒三角 stop
# 3    27     17      13   14

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

# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()

logits = my_net(x, n_classes=n_classes)

with tf.Session() as sess:
    # restore saved session
    checkpointer.restore(sess, './checkpoints/traffic_sign_model.ckpt-27')

    # predict on unseen images
    prediction = np.argmax( np.array( sess.run(logits) ) )

