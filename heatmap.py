# import the necessary packages
from gradcam import GradCAM

import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
import os
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="resnet50",
	choices=("vgg", "resnet50", "xception"),
	help="model to be used")
args = vars(ap.parse_args())

# initialize the model to be ResNet
model = keras.models.load_model('C:/Users/Robot1/Desktop/FODnew')

print(args["image"])
list = os.listdir(args["image"])
print(list)
print(type(list))
random.shuffle(list)
print(list)

for i in list[0:20]:
	# load the original image from disk (in OpenCV format) and then
	# resize the image to its target dimensions
	orig = cv2.imread(args["image"] + i)
	resized = cv2.resize(orig, (400, 400))
	# load the input image from disk (in Keras/TensorFlow format) and
	# preprocess it
	image = load_img(args["image"] + i, target_size=(400, 400))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# use the network to make predictions on the input image and find
	# the class label index with the largest corresponding probability
	preds = model.predict(image)
	score = preds[0]
	print("This image is %.2f percent good and %.2f percent bad."% (100 * (1 - score), 100 * score))

	if (score > 0.5):
		label = 'good'
		prob = score
	else: 
		label = 'bad'
		prob = 1 - score

	# initialize our gradient class activation map and build the heatmap
	cam = GradCAM(model, None, inner_model=model.get_layer('resnet50'), layerName='conv5_block3_out')
	heatmap = cam.compute_heatmap(image)
	# resize the resulting heatmap to the original input image dimensions
	# and then overlay heatmap on top of the image
	heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
	(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

	# draw the predicted label on the output image
	cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
	cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.8, (255, 255, 255), 2)
	# display the original image and resulting heatmap and output image
	# to our screen
	output = np.vstack([orig, heatmap, output])
	output = imutils.resize(output, height=700)
	cv2.imwrite("Heat" + i + ".jpg", output)
	#cv2.imshow("Output", output)
	#cv2.waitKey(0)
