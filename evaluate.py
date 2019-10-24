from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from model import model

_, (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape(*test_images.shape, 1)
test_images = test_images / 255.0

test_labels = to_categorical(test_labels, num_classes=10)

SAVED_MODEL = 'ToReport/Checkpoints/digits-cnn_best_weights.h5'
model.load_weights(SAVED_MODEL)

history_ev = model.evaluate(test_images, test_labels)
with open('ToReport/history_ev.txt', 'w') as f:
    f.write(history_ev.history)
