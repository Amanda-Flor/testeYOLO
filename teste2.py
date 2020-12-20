import cv2
import numpy as np
import time

from csv import DictWriter


video = cv2.VideoCapture(0)

#h, w = None, None

#with open('yoloDados/YoloNames.names') as f:
#    labels = [line.strip() for line in f]

#network = cv2.dnn.readNetFromDarknet('yoloDados/Yolov3.cfg')

print(cv2.__version__)

while True:
    conectado, frame = video.read()

    cv2.imshow('Video', frame)
    

    #codigo de detecção de objetos


    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

