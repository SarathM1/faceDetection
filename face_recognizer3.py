#!/usr/bin/python

import cv2, os
import numpy as np
from PIL import Image
import sys

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    images = []
    labels = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        image_pil = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image = np.array(image_pil, 'uint8')
        
        nbr = int(os.path.split(image_path)[1].split("_")[0].replace("s", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            #cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            #cv2.waitKey(50)
    return images, labels

path = './orl_faces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        box_text = "img = {},conf = {}".format(nbr_predicted, round(conf,3))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, box_text, (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 2)
        cv2.imshow("Recognizing Face", img)
        key = cv2.waitKey(20) & 0xff
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
