import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

data_root = ("C:/Users/Robot1/Desktop/FOD/git/224x224dataset")
TRAINING_DATA_DIR = str(data_root)
image_size = (400, 400)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode='binary',
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode='binary',
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)

base_model = keras.applications.ResNet50(
    weights = 'imagenet',  # Load weights pre-trained on ImageNet.
    input_shape = (400, 400, 3),
    pooling = 'max',
    include_top = False)  # Do not include the ImageNet classifier at the top.
    
base_model.trainable = False

inputs = keras.Input(shape=(400, 400, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
# x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Flatten()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy,
              metrics=[keras.metrics.BinaryAccuracy()])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="C:/Users/Robot1/Desktop/FODnew/",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
              
model.fit(train_ds, epochs=20, callbacks = [model_checkpoint_callback], validation_data=val_ds)

#os.system('shutdown -s')
