# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img, img_to_array
from keras import models
import json
import pandas as pd
import numpy as np
from PIL import Image
from keras import backend as K
import zipfile


def load_model():
    with zipfile.ZipFile('../dog_breed_clf_small.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    model = models.load_model('dog_breed_clf_small')
    # model._make_predict_function()
    session = K.get_session()
    return model, session


model, sess = load_model()

layer_outputs = [layer.output for layer in model.layers[:]] 
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

img = load_img('/home/ben/Downloads/IMG-1050.jpg', target_size=(512, 512))
img_array = img_to_array(img) / 255
img_batch = np.expand_dims(img_array, axis=0)

model.predict(img_batch).argmax(axis=1)

# + active=""
# model.summary()

# +
img_path = '/home/ben/Downloads/lab.jpg'

img = load_img(img_path, target_size=(512, 512))
img_array = img_to_array(img) / 255
img_batch = np.expand_dims(img_array, axis=0)
# -

model.predict(img_batch).argmax(axis=1)

model.summary()


# +
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block14_sepconv2_act', pred_index=None):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    
    img_pred = keras.preprocessing.image.load_img(img_path, target_size=(512, 512))
    img_pred = img_to_array(img_pred) / 255
    img_pred = np.expand_dims(img_pred, axis=0)

    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_pred)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()
    
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

import tensorflow as tf
import keras
import matplotlib.cm as cm
from IPython.display import Image, display
heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


save_and_display_gradcam(img_path, heatmap)

# -




