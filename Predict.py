import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

import datetime
import os
import cv2


modelPath = "C:/Users/Robot1/Desktop/FODnew/3ItemModel_v3"

testDataPath = "C:/Users/Robot1/Desktop/FODnew/DATA/Test"

model = tf.keras.models.load_model(modelPath)

image_size = (400, 400)

dir0 = os.listdir(testDataPath + "/good")
dir1 = os.listdir(testDataPath + "/bad")

f = open('C:/Users/Robot1/Desktop/FODnew/WrongPredictions_v2.txt', 'w')

results = len(dir0) + len(dir1)
total = len(dir0) + len(dir1)

Fp = 0
Fn = 0
Tp = 0
Tn = 0

for i in dir0:
    file = "C:/Users/Robot1/Desktop/FODnew/DATA/Test/good/" + str(i)
    img = keras.preprocessing.image.load_img(file, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predictions = predictions[0]
    if predictions < 0.5:
        results -= 1
        print(predictions, "BAD", i)
        string = "C:/Users/Robot1/Desktop/FODnew/DATA/Test/good/" + str(i) + "\n"
        f.write(string)
        Fn += 1
    else:
        print(predictions, "GOOD", i)
        Tp += 1

for i in dir1:
    file = "C:/Users/Robot1/Desktop/FODnew/DATA/Test/bad/" + str(i)
    img = keras.preprocessing.image.load_img(file, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predictions = predictions[0]
    if predictions < 0.5:
        print(predictions, "BAD", i)
        Tn += 1
    else:
        results -= 1
        print(predictions, "GOOD", i)
        string = "C:/Users/Robot1/Desktop/FODnew/DATA/Test/bad/" + str(i) + "\n"
        f.write(string)
        Fp += 1
        
print(f"Results: {results}/{total}    Accuracy: {float(results)/float(total)*100}%")
print(f"Fp = {Fp} Fn = {Fn} Tp = {Tp} Tn = {Tn}")
