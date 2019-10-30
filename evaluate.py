from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
from model import model

SAVED_MODEL = 'ToReport/Checkpoints/digits-cnn_best_weights.h5'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Матрица ошибок',
                          cmap=plt.cm.Blues):
    """
    Функция для печати матрицы ошибок.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Истинное значение')
    plt.xlabel('Предсказание модели')
    plt.show()


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ Функция печатает топ 6 ошибок классификации"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title(
                "Предсказание модели :{}\nИстинное значение :{}".format(pred_errors[error], obs_errors[error]))
            n += 1
    plt.show()


# Тестовая выборка:
_, (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape(*test_images.shape, 1)
test_images = test_images / 255.0

test_labels = to_categorical(test_labels, num_classes=10)

# Обученная модель:
model.load_weights(SAVED_MODEL)

## Матрица ошибок:
Y_pred = model.predict(test_images)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(test_labels, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))

## Топ 6 ошибок:
# ошибки:
errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_true_errors = test_images[errors]

# Вероятности:
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)
# Вероятности истинных значений:
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
# Разности:
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Топ 6 ошибок
sorted_delta_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_delta_errors[-6:]
display_errors(most_important_errors, X_true_errors, Y_pred_classes_errors, Y_true_errors)
