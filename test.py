from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from model import model
from sklearn.metrics import confusion_matrix
import itertools
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

SAVED_MODEL = 'ToReport/Checkpoints/digits-cnn_best_weights.h5'

_, (test_images, test_labels) = mnist.load_data()

N = len(test_images)
k = np.random.randint(N)

test_img = test_images[k].reshape(1, *test_images[k].shape, 1)
test_img = test_img / 255.0

model.load_weights(SAVED_MODEL)

pred = model.predict(test_img)
io.imshow(test_images[k])
plt.title(f'True label: {test_labels[k]}, Predicted: {np.argmax(pred)}')
plt.show()
