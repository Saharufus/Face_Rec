import cv2 as cv
import numpy as np
from preprocessing import load_all_from_path, rescale_resize
from tensorflow.keras.models import load_model
from model_building import DistLayer


def identify_loop(name):
    model = load_model('twin_model.h5', custom_objects={'DistLayer': DistLayer})
    cap = cv.VideoCapture(0)
    start_frame = 180
    voters = load_all_from_path(name)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[start_frame:start_frame + 250, start_frame:start_frame + 250]
        cv.imshow('im', frame)

        if cv.waitKey(1) & 0XFF == ord('i'):
            print('Identifying...')
            image = rescale_resize(frame)
            image_set = np.array([image for i in range(len(voters))])
            pred = model.predict([voters, image_set]) > 0.5
            print(pred.mean())
            if pred.mean() >= 0.7:
                print('Same person')
            else:
                print('Get out of here')

        if cv.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    identify_loop('Sahar')
