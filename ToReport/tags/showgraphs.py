import pandas as pd
import os
import matplotlib.pyplot as plt


def display_graphs(train, val, Title):
    """ Функция печатает графики ошибки/метрики качества в координатах значение/№ эпохи"""
    plt.plot(train['Step'], train['Value'], label='Обучение')
    plt.plot(val['Step'], val['Value'], label='Валидация')
    plt.legend()
    plt.grid()
    plt.title(Title)
    plt.ylabel(Title)
    plt.xlabel('Эпоха')
    plt.show()


# метрика качества на трейне и валидации:
train_acc = pd.read_csv('run-train-tag-epoch_accuracy.csv')
val_acc = pd.read_csv('run-validation-tag-epoch_accuracy.csv')

# ошибка на трейне и валидации:
train_loss = pd.read_csv('run-train-tag-epoch_loss.csv')
val_loss = pd.read_csv('run-validation-tag-epoch_loss.csv')

display_graphs(train_acc, val_acc, 'Значение метрики качества')
display_graphs(train_loss, val_loss, 'Значение ошибки')
