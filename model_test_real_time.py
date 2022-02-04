import cv2 as cv
import numpy as np
from preprocessing import load_all_from_path, rescale_resize
from tensorflow.keras.models import load_model
from model_building import DistLayer


def identify_loop(name, model_name, voters_threshold=0.5, mean_voters_threshold=0.8):
    """starts a loop for identifying a person from folder
    :param name: the name of the person to identify (make sure to set voters for this person)
    :param model_name: the name of the model to load
    :param voters_threshold: the threshold to set for each voter
    :param mean_voters_threshold: the threshold to set for the mean of all the voters"""
    # loads the model with the custom layer DistLayer
    model = load_model(model_name, custom_objects={'DistLayer': DistLayer})
    # opens the cam
    cap = cv.VideoCapture(0)
    start_frame = 180
    voters = load_all_from_path(name)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[start_frame:start_frame + 250, start_frame:start_frame + 250]
        cv.imshow('im', frame)

        if cv.waitKey(1) & 0XFF == ord('i'):
            # starting to identify
            print('Identifying...')
            image = rescale_resize(frame)
            # creating set if images with the length of the voters
            image_set = np.array([image for voter in voters])
            # predicting for every voter
            pred = model.predict([voters, image_set])
            pred_conf = (pred > voters_threshold).mean()
            print(pred_conf)
            # print the decision
            if pred_conf >= mean_voters_threshold:
                print('Same person')
            else:
                print('Get out of here')

        # close cam
        if cv.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    identify_loop('Orelle', 'twin_model.h5', 0.8, 0.8)
