import cv2
import os
import uuid
from time import sleep


def create_voters(name):
    """
    create a voter folder for a person
    :param name: name of person
    """
    try:
        os.makedirs(name)
    except FileExistsError:
        pass

    cap = cv2.VideoCapture(0)
    start_frame = 180
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[start_frame:start_frame + 250, start_frame:start_frame + 250]
        cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0XFF == ord('v'):
            print('Taking pictures...')
            for i in range(20):
                r, f = cap.read()
                f = f[start_frame:start_frame + 250, start_frame:start_frame + 250]
                imname = os.path.join(name, f'{uuid.uuid1()}.jpg')
                cv2.imwrite(imname, f)
                sleep(0.1)
            print('Done')

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_voters('Sahar')
