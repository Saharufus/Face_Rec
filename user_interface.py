import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from model_building import DistLayer
import numpy as np
from preprocessing import load_all_from_path, rescale_resize
import cv2
import time

st.set_page_config(page_title="Face Recognition", page_icon="📸️", layout='centered', initial_sidebar_state='auto')

st.markdown("<h1 style='text-align: center; color: gray;'>Face Recognition 📸</h1>", unsafe_allow_html=True)
model = load_model('FR_v1.h5', custom_objects={'DistLayer': DistLayer})

voter = st.selectbox('Who are you?', ['No one', 'Noam', 'Sahar', 'Juli'])

if voter in ['Noam','Sahar','Juli']:

    cap = cv2.VideoCapture(0)
    start_frame_x = 380
    start_frame_y = 150
    window = st.image([])
    button = st.button('Click here to take the picture!')

    while not button:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame[start_frame_y:start_frame_y + 250, start_frame_x:start_frame_x + 250], cv2.COLOR_BGR2RGB)
        window.image(frame, width = 400)

    ret, frame = cap.read()
    img = frame[start_frame_y:start_frame_y + 250, start_frame_x:start_frame_x + 250]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rescale_resize(img)

    st.image(img,width = 400)


    voters = load_all_from_path('voting_' + voter.lower())
    img_set = np.array([img for i in voters])


    with st.spinner(text='Please wait.. ⏱'):
        prediction = (model.predict([voters, img_set]) >= .75)
        st.success('Done')

    if prediction.mean() > 0.8:
        st.success(f"Yes, you're right, you are {voter} 🏆")
        st.balloons()

    else:
        st.error(f'You are not {voter}! You were lying ☠☠☠☠️!!!')

