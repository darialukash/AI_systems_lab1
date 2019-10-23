from sklearn.model_selection import train_test_split
# from keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import os

from model import model  # , learning_rate_reduction

RANDOM_SEED = 3
EPOCHS = 30  # Turn epochs to 30 to get 0.9967 accuracy
BATCH_SIZE = 64
MODEL_PATH = 'ToReport'
NAME = "digits-cnn"

checkpoints = []
if not os.path.exists('ToReport/Checkpoints/'):
    os.makedirs('ToReport/Checkpoints/')

checkpoints.append(ModelCheckpoint(f'ToReport/Checkpoints/{NAME}_best_weights.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True,
                                   save_weights_only=True, mode='auto'))
checkpoints.append(TensorBoard(log_dir='ToReport/Checkpoints/./logs', histogram_freq=1,
                               write_graph=True, write_images=False, embeddings_freq=1,
                               embeddings_layer_names=None, embeddings_metadata=None))
checkpoints.append(ReduceLROnPlateau(monitor='val_loss',
                                     patience=3,
                                     verbose=1,
                                     factor=0.3,
                                     min_lr=0.00001))

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)
# print(train_labels.shape)

train_images = train_images.reshape(*train_images.shape, 1)
test_images = test_images.reshape(*test_images.shape, 1)

# print(train_images.shape)
# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels, num_classes=10)

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1,
                                                  random_state=RANDOM_SEED)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# data augmentation:
datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                              epochs=EPOCHS, validation_data=(X_val, Y_val),
                              verbose=1, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=checkpoints)

with open('ToReport/history.json', 'w') as outfile:
    json.dump(history.history, outfile)
