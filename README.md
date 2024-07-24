# siamese-network-contrastive-loss
Данный проект нацелен на определение наиболее похожего изображения для картинок-заполнителей. Исходный код с комментариями представлен в файле siamese_network_contrastive_loss.ipynb, набор данных - в файле archive_ai_0.zip
# Оценка сходства изображений с использованием сиамской сети с контрастивной функцией потерь
**Дата создания:** 2024/05/08<br>
**Исходник:** https://habr.com/ru/companies/jetinfosystems/articles/465279/<br>
**Описание:** Обучение сиамской сети на созданном наборе данных (85 классов по 14 картинок) с использованием ResNet50, функция потерь - Contrastive Loss<br>
**Примечание:** Картинки в формате JPEG, цветовая модель RGB

## Импорт библиотек

!apt install unzip </br>
import os</br>
import shutil</br>
import pathlib</br>
import cv2</br>
import re</br>
import numpy as np</br>
from PIL import Image</br>

from tensorflow import keras</br>
from sklearn.model_selection import train_test_split</br>
from keras import backend as K</br>
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, GlobalMaxPooling2D, Activation</br>
from keras.models import Sequential, Model, load_model</br>
from keras.optimizers import RMSprop</br>
from tensorflow.python.keras.backend import expand_dims</br>
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score</br>
from keras.applications import resnet</br>
from keras import regularizers</br>
import random</br>
import matplotlib.pyplot as plt</br>

## Загрузка набора данных для обучения и предварительная обработка
Набор данных будет состоять из 1190 картинок, то есть 85 классов по 14 аугментированных картинок.

## Функция генерации данных
В этом блоке будет происходить считывание изображения, получение нового размера, инициализация массива numpy в форме [total_sample, no_of_pairs, dim1, dim2] и создание подлинных и ложных пар.

## Генерация подлинных и ложных пар
Размер изображения будет total_sample_size = 1190

## Создание и обучение нейронной сети

75 % пар пойдет на обучение, а 25 % — на тестирование. В качестве базовой сети будет использоваться свёрточная нейросеть ResNet-50 для извлечения свойств.

## Преобразование размерности входных изображений
input_dim = (150, 140, 3)

feat_vecs_a и feat_vecs_b — это векторы свойств пары изображений, передадим их функции энергии для вычисления дистанции между ними. А в качестве функции энергии воспользуемся евклидовым расстоянием.

## Обучение сети

Зададим число эпох, применим свойство RMS для оптимизации и объявим модель, а также определим функцию потерь contrastive_loss function и скомпилируем модель.

![image](https://github.com/user-attachments/assets/94c417a8-683b-4b42-b824-8d217e9fd73f)</br>

![image](https://github.com/user-attachments/assets/1c2db639-c94c-4bab-96af-df18ca0d6695)</br>

## Подбор порога по F1-мере
![image](https://github.com/user-attachments/assets/f836b1ee-3a07-4664-9cdf-b116f4bfdffb)</br>

## Тестирование

![image](https://github.com/user-attachments/assets/1aebb41e-8d87-4f15-b8a9-06b0d8f1f0f3)</br>

## Создание интерфейса Gradio

![image](https://github.com/user-attachments/assets/7b68c93a-736e-47dd-b0f4-bf1a67c54853)</br>
![!!!!!!!!!!!!!!!!!!!!!!!!](https://github.com/user-attachments/assets/c4bbdff3-6f98-4f9e-9348-f0e5a3e0ecc8)</br>
![!!!!!!!](https://github.com/user-attachments/assets/8a5cff7c-4464-4c03-a70b-970016283293)</br>



