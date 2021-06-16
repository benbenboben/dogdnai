from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img, img_to_array
from keras import models
import json
import pandas as pd
import numpy as np
from PIL import Image
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import zipfile
from matplotlib import cm

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)


def gradcam(img_path, model, last_conv_layer_name='block14_sepconv2_act', pred_index=None):
    img = my_load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    img_pred = my_load_img(img_path, target_size=(512, 512))

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
    superimposed_img = jet_heatmap * 0.5 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


@st.cache(allow_output_mutation=True)
def load_model():
    with zipfile.ZipFile('dog_breed_clf_small.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    model = models.load_model('dog_breed_clf_small')
    session = K.get_session()
    return model, session


@st.cache
def load_breeds():
    df = pd.DataFrame(pd.Series(json.load(open('class_index_map.json', 'r'))))
    breeds = df.index
    breeds = [' '.join(i.split('-')[1:]).replace('_', ' ').title() for i in breeds]
    return breeds


def make_prediction_table(yhat):
    breeds = load_breeds()
    df = pd.DataFrame([{'Breed': i, 'Percentage': j, 'Index': z} for z, (i, j) in enumerate(zip(breeds, yhat.ravel()))])
    # df = df.set_index('Breed')
    df['Percentage'] = (df['Percentage'] * 100).round(2)
    df = df[df['Percentage'] > 0]
    return df.sort_values('Percentage', ascending=False).head(5)


st.set_page_config(page_title='🤖🐕', page_icon = '🤖🐕', initial_sidebar_state = 'auto')
st.title('Dog DN(AI)')
st.write('Predict your dog\'s breed with computer vision')


@st.cache(max_entries=8)
def my_load_img(img_path, target_size=None):
    if target_size is None:
        return load_img(img_path)
    else:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch


buffer = st.file_uploader(
    'Upload a picture of your dog to see what he or she might be (well-cropped close-ups work best)!'
)

temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    orig_image = my_load_img(temp_file.name)
    st.image(orig_image)
    img_batch = my_load_img(temp_file.name, target_size=(512, 512))
    model, session = load_model()
    probas = model.predict(img_batch)
    table = make_prediction_table(probas)
    st.table(table[['Breed', 'Percentage']].set_index('Breed'))
    selected_breed = st.selectbox('Select breed to see what features stood out to the model', table['Breed'].values)
    st.image(gradcam(temp_file.name, model, pred_index=table[table['Breed'] == selected_breed]['Index'].values[0]))
