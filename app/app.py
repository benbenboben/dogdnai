from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img, img_to_array
from keras import models
import json
import pandas as pd
import numpy as np
from PIL import Image
from keras import backend as K
import zipfile

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    with zipfile.ZipFile('dog_breed_clf_small.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    model = models.load_model('dog_breed_clf_small')
    # model._make_predict_function()
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
    df = pd.DataFrame([{'Breed': i, 'Percentage': j} for i, j in zip(breeds, yhat.ravel())])
    df = df.set_index('Breed')
    df['Percentage'] = (df['Percentage'] * 100).round(2)
    return df.sort_values('Percentage', ascending=False).head(5)


st.set_page_config(page_title='🤖🐕', page_icon = '🤖🐕', initial_sidebar_state = 'auto')
st.title('Dog DN(AI)')
st.write('Predicting your dog breed with computer vision')

# col1, col2 = st.beta_columns(2)

buffer = st.file_uploader('Upload a picture of your dog to see what he or she might be!')
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    orig_image = load_img(temp_file.name)
    st.image(orig_image)
    img = load_img(temp_file.name, target_size=(512, 512))
    img_array = img_to_array(img) / 255
    img_batch = np.expand_dims(img_array, axis=0)
    model, session = load_model()
    probas = model.predict(img_batch)
    st.table(make_prediction_table(probas))
