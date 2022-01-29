import cv2
import numpy as np
from preprocessing import load_from_path, rescale_resize, load_data
from tensorflow.keras.models import load_model
from model_building import DistLayer
import uuid
import os


def identify_loop(name):
    model = load_model('twin_model.h5', custom_objects={'DistLayer': DistLayer})
    cap = cv2.VideoCapture(0)
    start_frame = 180
    num_voters = 120
    voters = load_from_path(name, num_voters)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[start_frame:start_frame + 250, start_frame:start_frame + 250]
        cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0XFF == ord('i'):
            print('Identifying...')
            image = rescale_resize(frame)
            image_set = np.array([image for i in range(num_voters)])
            pred = model.predict([voters, image_set]) > 0.5
            print(pred.mean())
            if pred.mean() >= 0.7:
                print('Same person')
            else:
                print('Get out of here')

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    identify_loop('Orelle')