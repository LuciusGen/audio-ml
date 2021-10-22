#!/usr/bin/env python
# coding: utf-8

# # Постановка задачи
# Виртуальные ассистенты, устройства Internet-of-Things все больше входят в жизнь современного человека. Они не только помогают автоматизировать поисковые запросы, распознают лица и речь, выполняют простейшие команды, но и учатся вести мониторинг состояния здоровья, детектировать различные ситуации, информировать о важных для пользователя событиях.
# 
# Для того, чтобы виртуальные ассистенты реагировали только на голос человека, присутствующего перед устройством, и не принимали во внимание речь из телевизора, радио, а также синтезированную, воспроизводимую роботами и другую, звучащую из динамиков, необходимы детекторы “живого” голоса.
# 
# Данная задача посвящена детектированию наличия “живого” голоса (класс 1) и его отделению от синтетического/конвертированного/перезаписанного голоса (класс 2).
# 
# Предлагается разработать систему с использованием элементов машинного
# обучения, которая обучается на заданной обучающей базе аудиозаписей и должна быть протестирована на тестовой базе аудиозаписей.
# 
# # Исходные данные
# 1. Обучающая база данных:
# a. Ссылка на базу: training_data
# b. База имеет метки human (класс №1) и spoof (класс №2)
# 
# 2. Тестовая база данных:
# a. Ссылка на тестовую базу: testing_data
# b. База не имеет меток правильных ответов
# 
# # Форма представления результата
# 
# 1. Текстовый файл с ответами системы на тестовой базе в формате:
# <имя файла>, <score>
# где score - значение выхода системы детектирования. Чем выше score, тем больше уверенность системы в том, что в файле был записан “живой” голос.
# 
# 2. Любая принципиально важная часть исходных кодов, по которой можно понять
# алгоритм решения задачи
# 
# 3. Собственные комментарии и замечания (по желанию)
# 
# Список литературы (опционально)
# 
# Основной список
# 
# https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2279.pdf
#     
# http://www.apsipa.org/proceedings/2018/pdfs/0001047.pdf
# 
#    https://pdfs.semanticscholar.org/6a7b/88c8dc37850f8ffe48dcf7d839c6f0d47873.pdf



conda install -c conda-forge librosa


import librosa
import matplotlib.pyplot as plt


# В качестве примера обработаем несколько файлов из тренировочного набора данных.

import matplotlib.pyplot as plt
import librosa.display

x_h, sr_h = librosa.load("Training_Data\\human\\human_00002.wav")

X_h = librosa.stft(x_h)
Xdb_h = librosa.amplitude_to_db(abs(X_h))

librosa.display.specshow(Xdb_h, sr=sr_h, x_axis='time', y_axis='hz')
plt.colorbar()



import matplotlib.pyplot as plt
import librosa.display

x_s, sr_s = librosa.load("Training_Data\\spoof\\spoof_00002.wav")

X_s = librosa.stft(x_s)
Xdb_s = librosa.amplitude_to_db(abs(X_s))

librosa.display.specshow(Xdb_s, sr=sr_s, x_axis='time', y_axis='hz')
plt.colorbar()


# Для того, чтобы работать с аудиофайлами, необходимо выбрать ключевые особенности.
# В качестве таковых были выбраны следующие особенности:
# 
# Разложение временного ряда аудио на гармонические и ударные компоненты(librosa.effects.hpss)
# 
# MFCC гармонической состовляющей аудио(количетво возврощаемых mfcc == 10)(librosa.feature.mfcc)
# 
# Отслеживание битов гармонической состовляющей аудио(librosa.beat.beat_track)
# 
# Нахождение частоты спада исходного аудио(librosa.feature.spectral_rolloff)
# 
# Вычисление спектрального контраста гармонической состовляющей(librosa.feature.spectral_contrast)
# 
# Частота перехода через 0 временного ряда гармонической состовляющей аудиозаписиlibrosa.feature.zero_crossing_rate
# 
# Спектральнаая полоса частот librosa.feature.spectral_bandwidth
# 
# Спектральный контраст librosa.feature.spectral_contrast
# 
# Спектральный центроид librosa.feature.spectral_centroid
# 
# У всех вычисленных характеристик было вычислено выборочное среднее и выборочное стандартное отклонение.


import librosa
import numpy as np
import librosa.display
import scipy
from scipy import stats

def get_features(wav_file_path):
    features = list()
    x, sr = librosa.load(wav_file_path, sr=None)
    n_mfc, n_cens = 10, 12

    x_harmonic, x_percussive = librosa.effects.hpss(x)
    
    mfc = librosa.feature.mfcc(y=x_harmonic, sr=sr, n_mfcc=n_mfc)
    tempo, _ = librosa.beat.beat_track(y=x_harmonic, sr=sr)
    sp_contrast = librosa.feature.spectral_contrast(y=x_harmonic, sr=sr)
    
    zcr = librosa.feature.zero_crossing_rate(x_harmonic)
    cens = librosa.feature.spectral_centroid(y=x, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=x_harmonic, sr=sr)
    sp_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=x_harmonic, sr=sr)
    
    data = np.concatenate((np.mean(sp_contrast,axis=1), np.std(sp_contrast,axis=1),
                           [np.mean(sp_rolloff), np.std(sp_rolloff)], [tempo],
                           [np.mean(zcr), np.std(zcr)], [np.mean(contrast), np.std(contrast)],
                           [np.mean(bandwidth), np.std(bandwidth)],
                           np.mean(mfc, axis=1)[0: n_mfc], np.std(mfc, axis=1)[0: n_mfc],
                           np.mean(cens, axis=1)[0: n_cens], np.std(cens, axis=1)[0: n_cens]), axis=0)
    
    features.extend(list(data))
  
    return features


# Далее вычисляются характеристики тренировочных и тестовых данных и сохраняются в временные файлы, для избежания повторного вычисления.

import os

def get_data(way):
    data = list()
    
    for subdir, _, files in os.walk(way):
        for file in files:
            data.append(get_features(os.path.join(subdir, file)))
    return data

h_tr_data_way = 'Training_Data\\human'
s_tr_data_way = 'Training_Data\\spoof'

h_features = get_data(h_tr_data_way)
s_features = get_data(s_tr_data_way)


# In[6]:


def data_save(data_list: list, files_name: str, class_type: str):
    x_f = open(files_name + "_x.txt", 'w')
    y_f = open(files_name + "_y.txt", 'w')
    
    for i in data_list:
        x_f.write(' '.join([str(j) for j in i]) + '\n')
        y_f.write(class_type + '\n')
    
    x_f.close()
    y_f.close()

data_save(h_features, 'human', '1')
data_save(s_features, 'spoof', '2')


def get_test_data(way):
    data = list()
    list_files = list()
    
    for subdir, _, files in os.walk(way):
        list_files.extend(files)
        for file in files:
            data.append(get_features(os.path.join(subdir, file)))
    return (data, files)

test_data_way = 'Testing_Data'

test_data, list_files = get_test_data(test_data_way)



data_save(test_data, 'test', '0')
data_save(list_files, 'test_f_names', '0')


# В качестве модели используется градиентный бустинг реализованный в библиотеке CatBoostClassifier.


from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
estimators, iters = 2000, 2000

X_train = h_features + s_features
Y_train = [1] * len(h_features) + [2] * len(s_features)
boost = CatBoostClassifier(n_estimators=estimators)

score = cross_val_score(boost, X_train, Y_train, cv =5, 
                        scoring=make_scorer(accuracy_score, greater_is_better=False))
print(score)


boost.fit(X_train, Y_train)
y_score = boost.predict_proba(test_data)



f = open('res_tmp.txt', 'w')
for i in range(len(list_files)):
    f.write(list_files[i] + " " + str(y_score[i][0]) + '\n')
f.close()


def load_data(filename: str):
    file = open(filename, 'r')
    data = []
    for line in file:
        data.append([float(i) for i in line.split()])
    file.close()
    return data

h_features = load_data('human_x.txt')
s_features = load_data('spoof_x.txt')
test_data = load_data('test_x.txt')
file = open('test_f_names_x_0.txt', 'r')
list_files = []
for line in file:
    list_files.append(''.join(line.split()))
file.close()

conda install -c conda-forge lightgbm


from lightgbm import LGBMClassifier

estimators = 2000

X_train = h_features + s_features
Y_train = [1] * len(h_features) + [2] * len(s_features)

model = LGBMClassifier(n_estimators=estimators)

score = cross_val_score(model, X_train, Y_train, cv =5, 
                        scoring=make_scorer(accuracy_score, greater_is_better=False))
print(score)


X_train = h_features + s_features
Y_train = [1] * len(h_features) + [2] * len(s_features)

model = LGBMClassifier(n_estimators=estimators)
model.fit(X_train, Y_train)
y_score = model.predict_proba(test_data)

f = open('res_lgbm.txt', 'w')
for i in range(len(list_files)):
    f.write(list_files[i] + " " + str(y_score[i][0]) + '\n')
f.close()
