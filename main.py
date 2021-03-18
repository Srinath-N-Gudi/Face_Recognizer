import cv2
import os
import time
faceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
address = "https://25.192.242.202:5028/video"
video = cv2.VideoCapture(0)
video.open(address)

name = input("Enter your name: ")
time.sleep(3)
print("Started")
if not os.path.exists(name):    
    os.mkdir(name)
os.chdir(os.getcwd()+f"/{name}")
# print("Please wait")
i = 1
while True:
    ret, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceData.detectMultiScale(gray_img, 1.1, 4)
    for (x, y, w, h) in faces:
        
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        if i < 11:
            cv2.imwrite(str(i)+".png", face)
            i=i+1
        else:
            exit()
    # cv2.imshow("Window",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows
