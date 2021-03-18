import cv2
import os
from PIL import Image
import numpy as np
import pickle

current_id = 0
x_train = []
y_label = []
label_ids = {}

face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

Base_dir = os.getcwd()
image_dir = os.path.join(Base_dir, "Traning Data")

# Setting up the face recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            # Getting the label of the image
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)    
            # Giving the image their label form the above data
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            # Getting the images and then converting them into a numpy array
            image = Image.open(path).convert('L')
            image = image.resize((550, 550), Image.ANTIALIAS)
            image_array = np.array(image, "uint8")
            
            # Getting the region of interest that is the face only
            faces = face_data.detectMultiScale(image_array, 1.5, 5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_label.append(id_)
# print(x_train, y_label)
with open("labels.pkl", 'wb') as file:
    pickle.dump(label_ids, file)
print("The labels have been dumped, training started successfully.")
print("Please wait...")
recognizer.train(x_train, np.array(y_label))
recognizer.save("faces.yml")
