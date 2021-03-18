import cv2
import pickle
faceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('faces.yml')
label= {}
with open('labels.pkl', 'rb') as file:
    label_ids = pickle.load(file)
    label = {v:k for k,v in label_ids.items()}
while True:
    ret, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceData.detectMultiScale(gray_img, 1.1, 4)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = gray_img[y:y+h, x:x+w]


        id_, conf = recognizer.predict(face_gray)
        if conf >= 45:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(frame, label[id_], (x,y-5), cv2.FONT_ITALIC, 5 , (255, 0, 0))
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)
            cv2.putText(frame, "Unknown", (x,y-5), cv2.FONT_ITALIC, 5 , (255, 0, 0))



    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows
