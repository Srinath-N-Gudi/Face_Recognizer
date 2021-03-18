import cv2
import pickle
import EasyCode

faceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
address = "https://10.162.241.7:5028/video"
video = cv2.VideoCapture(0)
video.open(address)

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
        
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_gray = gray_img[y:y+h, x:x+w]


        id_, conf = recognizer.predict(face_gray)
        # print(conf)
        if conf >= 45:
            # print(conf)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(frame, label[id_], (x,y-5), cv2.FONT_ITALIC, 5 , (255, 0, 0))
            # EasyCode.speak("Hello "+ label[id_])
##            print(label[id_])
        else:

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)
            cv2.putText(frame, "Unknown", (x,y-5), cv2.FONT_ITALIC, 5 , (255, 0, 0))



    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows
