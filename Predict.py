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


modelPath = "C:/Users/Robot1/Desktop/FODnew/"

testDataPath = "C:/Users/Robot1/Desktop/FODnew/testData"

model = tf.keras.models.load_model(modelPath)#,custom_objects={'KerasLayer':hub.KerasLayer})

image_size = (400, 400)

dir = os.listdir(testDataPath)

for i in dir:
    file = "C:/Users/Robot1/Desktop/FODnew/testData/" + str(i)
    img = keras.preprocessing.image.load_img(file, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predictions = predictions[0]
    print(predictions)
    if predictions > 0.5:
        print("BAD")
    else:
        print("GOOD")
