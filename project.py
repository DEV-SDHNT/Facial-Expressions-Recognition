import cv2
import joblib
import numpy as np

print("Loading model ..")
model=joblib.load('fer_model.pkl')
print("Model loaded .")

classes=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preProcessedFrame(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(48,48))
    flattened=np.stack(resized)
    # flattened=resized.flatten()
    return flattened

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0)
while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        processedFace=preProcessedFrame(face)
        pred=model.predict(processedFace.reshape(1,-1))
        print(pred ,classes[int(pred)])
    cv2.imshow("face",frame)
    cv2.waitKey(1)
