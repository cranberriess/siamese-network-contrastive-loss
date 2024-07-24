# siamese-network-contrastive-loss
Данный проект нацелен на определение наиболее похожего изображения для картинок-заполнителей. Исходный код с комментариями представлен в файле siamese_network_contrastive_loss.ipynb, набор данных - в файле archive_ai_0.zip
# Оценка сходства изображений с использованием сиамской сети с контрастивной функцией потерь
**Дата создания:** 2024/05/08<br>
**Исходник:** https://habr.com/ru/companies/jetinfosystems/articles/465279/<br>
**Описание:** Обучение сиамской сети на созданном наборе данных (85 классов по 14 картинок) с использованием ResNet50, функция потерь - Contrastive Loss<br>
**Примечание:** Картинки в формате JPEG, цветовая модель RGB

##Импорт библиотек

!apt install unzip 
import os
import shutil
import pathlib
import cv2
import re
import numpy as np
from PIL import Image

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, GlobalMaxPooling2D, Activation
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop
from tensorflow.python.keras.backend import expand_dims
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
from keras.applications import resnet
from keras import regularizers
import random
import matplotlib.pyplot as plt

##Загрузка набора данных для обучения

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

os.chdir('/content')

#!rm -rf "/content/archive_ai"

!unzip -u "/content/drive/MyDrive/archive_ai_0.zip"

for i in range(85): #переименование архива
  os.chdir('/content/archive_ai/augmented_image_' + str(i+1)+'/')
  path = pathlib.Path('/content/archive_ai/augmented_image_' + str(i+1)+'/')
  for j, path in enumerate(path.glob('*.jpg')):
    new_name = str(j+1) + '.jpg'
    path.rename(new_name)

##Предобработка картинок для обучения

path = "/content/archive_ai/augmented_image_9/1.jpg"

img = Image.open(path)

img

img_array = np.array(img)
img_array.shape

m = img_array[:,:,:]
m.shape

x = expand_dims(m, axis=0)
x.shape


x = expand_dims(x, axis=0)
x.shape

##Функция генерации данных

def get_data(total_sample_size):

    #считываем изображение
    image = cv2.imread('/content/archive_ai/augmented_image_' + str(1) + '/' + str(1) + '.jpg')

    #получаем новый размер
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0

    #инициализируем массив numpy в форме [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 3, dim1, dim2]) # 2 для пар
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(85):
        for j in range(int(total_sample_size/85)):
            ind1 = 0
            ind2 = 0

            #чтение изображений из одного каталога (подлинная пара)
            while ind1 == ind2:
                ind1 = np.random.randint(14)
                ind2 = np.random.randint(14)

            #читаем два изображения
            img1 = cv2.imread('/content/archive_ai/augmented_image_' + str(i+1) + '/' + str(ind1 + 1) + '.jpg')
            img1 = img1.reshape(3,150,140)

            img2 = cv2.imread('/content/archive_ai/augmented_image_' + str(i+1) + '/' + str(ind2 + 1) + '.jpg')
            img2 = img2.reshape(3,150,140)

            #сохраняем изображения в инициализированном массиве numpy
            x_geuine_pair[count, 0, :, :, :] = img1
            x_geuine_pair[count, 1, :, :, :] = img2

            #поскольку мы рисуем изображения из того же каталога, мы присваиваем метке значение 1. (подлинная пара)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 3, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size/14)):
        for j in range(14):

            #чтение изображений из другого каталога (пара imposite)
            while True:
                ind1 = np.random.randint(85)
                ind2 = np.random.randint(85)
                if ind1 != ind2:
                    break

            img1 = cv2.imread('/content/archive_ai/augmented_image_' + str(ind1+1) + '/' + str(j + 1) + '.jpg')
            img1 = img2.reshape(3,150,140)

            img2 = cv2.imread('/content/archive_ai/augmented_image_' + str(ind2+1) + '/' + str(j + 1) + '.jpg')
            img2 = img2.reshape(3,150,140)

            x_imposite_pair[count, 0, :, :, :] = img1
            x_imposite_pair[count, 1, :, :, :] = img2
            #поскольку мы рисуем изображения из другого каталога, мы присваиваем метке значение 0. (пара imposite)
            y_imposite[count] = 0
            count += 1

    #объединяем подлинные пары и ложную пару, чтобы получить все данные целиком
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

##Генерация подлинных и ложных пар

total_sample_size = 1190

X, Y = get_data(total_sample_size)


X.shape

Y.shape

## Создание и обучение нейронной сети


75 % пар пойдет на обучение, а 25 % — на тестирование

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

x_test.shape

y_test.shape

x_train.shape

y_train.shape

x = expand_dims(m, axis=0)
x.shape


x = expand_dims(x, axis=0)
x.shape


Определим базовую сеть — это будет свёрточная нейросеть для извлечения свойств

def build_base_network(input_shape):
    seq = Sequential()

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=input_shape, include_top=False,
    )
    seq.add(base_cnn)

    seq.add(GlobalMaxPooling2D())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dense(64, activation='relu'))
    seq.add(Dense(32, activation='relu'))

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return seq

#передадим пару изображений базовой сети, которая вернёт векторные представления, то есть векторы свойств

# Преобразование размерности входных изображений
input_dim = (150, 140, 3)

img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)

input_dim

feat_vecs_a и feat_vecs_b — это векторы свойств пары изображений, передадим их функции энергии для вычисления дистанции между ними. А в качестве функции энергии воспользуемся евклидовым расстоянием:

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

##Обучение сети

Зададим число эпох, применим свойство RMS для оптимизации и объявим модель:

epochs = 40
rms = RMSprop()

model = Model(inputs=[img_a, img_b], outputs=distance)

model.summary()

Определим функцию потерь contrastive_loss function и скомпилируем модель:

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() <  0.52203].mean()

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred <  0.52203, y_true.dtype)))

model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

img_1 = x_train[:, 0]
img_2 = x_train[:, 1]


img_1 = img_1.reshape(img_1.shape[0], 150, 140, 3).astype('float32')
img_2 = img_2.reshape(img_2.shape[0], 150, 140, 3).astype('float32')

x_train.shape

print(img_1.shape)
print(img_2.shape)

max(Y)

history = model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=256, verbose=2, epochs=epochs)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.grid()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, 'b', label='Training accuracy')
plt.plot(val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.grid()

model.save('sinet.h5')

# Получение предсказаний модели на тестовом наборе данных
y_pred = model.predict([np.transpose(x_test[:, 0], (0, 2, 3, 1)), np.transpose(x_test[:, 1], (0, 2, 3, 1))])

# точность и полнота для различных значений порога
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)

best_f1 = -1
best_threshold = None
f1_scores = []
thresholds_plot = []

# порог, при котором F1-мера максимальна
for i in range(len(thresholds)):
    threshold = thresholds[i]
    y_pred_thresholded = y_pred < threshold
    f1 = f1_score(y_test, y_pred_thresholded)
    f1_scores.append(f1)
    thresholds_plot.append(threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Best Threshold by maximizing F1 Score:", best_threshold)
print("Best F1 Score:", best_f1)

# построение графика
plt.plot(thresholds_plot, f1_scores)
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.grid()
plt.show()

x.shape

z = x_test[0, 0]
z.shape

x = expand_dims(z, axis=1)
x.shape


w = x_test[0, 1]
w.shape

y = expand_dims(w, axis=1)
y.shape

##Тестирование

model_new = keras.models.load_model('sinet.h5', compile=False)

target_label = 1
values = np.array(y_test[:,0])

target_index = values.tolist().index(target_label)
print(target_index)
print('target_index value : ',y_test[target_index])

img1 = (x_test[target_index, 0] * 255).astype(np.uint8)
img1 = img1.reshape(150,140,3)
print(img1.shape)
img1
plt.imshow(img1)
plt.show()

img2 = (x_test[target_index, 1] * 255).astype(np.uint8)
img2 = img2.reshape(150,140,3)
print(img2.shape)
img2
plt.imshow(img2)
plt.show()

x_test[target_index:target_index+1, 0].shape

pred = model.predict([np.transpose(x_test[target_index:target_index+1, 0], (0, 2, 3, 1)), np.transpose(x_test[target_index:target_index+1, 1], (0, 2, 3, 1))])
print(pred)
pred = pred < best_threshold
print('y_test[target_index]:',y_test[target_index,0]==True,' pred :',pred)

target_label = 0
values = np.array(y_test[:,0])

target_index = values.tolist().index(target_label)
print(target_index)
print('target_index value : ',y_test[target_index])

img1 = (x_test[target_index, 0] * 255).astype(np.uint8)
img1 = img1.reshape(150,140,3)
print(img1.shape)
img1
plt.imshow(img1)
plt.show()

img2 = (x_test[target_index, 1] * 255).astype(np.uint8)
img2 = img2.reshape(150,140,3)
print(img2.shape)
img2
plt.imshow(img2)
plt.show()

x_test[target_index:target_index+1, 0].shape

pred = model.predict([np.transpose(x_test[target_index:target_index+1, 0], (0, 2, 3, 1)), np.transpose(x_test[target_index:target_index+1, 1], (0, 2, 3, 1))])
print(pred)
pred = pred < best_threshold
print('y_test[target_index]:',y_test[target_index,0]==True,' pred :',pred)

y = expand_dims(w, axis=1)
y.shape

# Определение количества пар картинок, которые вы хотите вывести
num_pairs_to_display = 4

for _ in range(num_pairs_to_display):
    # Генерация случайного индекса
    index = random.randint(0, len(y_test) - 1)

    # Отображение пары картинок
    img1 = (x_test[index, 0] * 255).astype(np.uint8)
    img1 = img1.reshape(150, 140, 3)

    img2 = (x_test[index, 1] * 255).astype(np.uint8)
    img2 = img2.reshape(150, 140, 3)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Image 2')

    plt.show()

    # Вычисление pred и ответа
    pred = model.predict([np.transpose(x_test[index:index+1, 0], (0, 2, 3, 1)), np.transpose(x_test[index:index+1, 1], (0, 2, 3, 1))])
    pred_label = "Картинки похожи" if pred < best_threshold else "Картинки не похожи"
    print("Prediction:", pred_label,  pred)
    print("Actual Label:", "Картинки похожи" if y_test[index, 0] == True else "Картинки не похожи")
    print("-" * 50)


##Создание интерфейса Gradio

!pip install --upgrade gradio

import gradio as gr

def resize_image(image):
    if image.shape[:3] != (150, 140, 3):
        image = cv2.resize(image, (140, 150))
    return image

def find_most_similar_image(user_image):
    img = resize_image(user_image)
    # Предобработка загруженного изображения
    img = user_image.reshape((1, 150, 140, 3)).astype('float32') / 255  # Приведение к формату и нормализация

    most_similar_image = None
    most_similar_index = None
    max_similarity = float('inf')
    index = None

    for i in range(len(x_test)):
        # Предобработка изображения из тестового набора данных
        test_image = x_test[i, 0].reshape((1, 150, 140, 3))

        # Предсказание сходства
        pred = model.predict([test_image, img])

        # Если сходство ниже максимального и превышает порог, обновляем наиболее похожее изображение и его индекс
        if pred < max_similarity and pred < best_threshold:
            most_similar_image = x_test[i, 0]
            most_similar_index = i
            max_similarity = pred

    test2 = x_test[most_similar_index, 0].reshape((1, 150, 140, 3))
    pred2 = model.predict([test2, img])

    # Вывод наиболее похожего изображения
    if pred2 < best_threshold:
        print("Наиболее похожее изображение:", pred2)
        plt.imshow(test2.reshape((150, 140, 3)))
        plt.axis('off')
        plt.show()
        return "Наиболее похожее изображение:", pred2, plt

iface = gr.Interface(fn=find_most_similar_image,
                     inputs="image",
                     outputs=["text", "text", "plot"],
                     title="Поиск наиболее похожего изображения",
                     description="Загрузите изображение, чтобы найти наиболее похожее изображение из набора данных.")

iface.launch()

