from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

from model import model

RANDOM_SEED = 3
EPOCHS = 30
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

# База данных цифр MNIST:
(train_images, train_labels), _ = mnist.load_data()

# print(train_images.shape)
# print(train_labels.shape)

train_images = train_images.reshape(*train_images.shape, 1)


# print(train_images.shape)
# Нормализация
train_images = train_images / 255.0

train_labels = to_categorical(train_labels, num_classes=10)

# Разделение обучающей выборки базы данных на обучение и валидацию
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1,
                                                  random_state=RANDOM_SEED)
# аугментация изображений
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=False,
    vertical_flip=False)
datagen.fit(X_train)

# Обучение
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                              epochs=EPOCHS, validation_data=(X_val, Y_val),
                              verbose=1, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=checkpoints)
