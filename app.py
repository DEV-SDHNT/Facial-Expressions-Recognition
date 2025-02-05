import cv2
import joblib
import numpy as np
from flask import Flask,render_template,Response

app=Flask(__name__)

# Load the trained Model .
model=joblib.load('fer_model.pkl')


classes=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


def preProcessedFrame(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(48,48))
    flattened=np.stack(resized)
    return flattened

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def Camera():
    cam=cv2.VideoCapture(0)
    while True:
        success,frame=cam.read()
        if not success:
            break
        else:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
            for (x,y,w,h) in faces:
                face=frame[y:y+h,x:x+w]
                frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                processedFace=preProcessedFrame(face)
                pred=model.predict(processedFace.reshape(1,-1))
                cv2.putText(frame,classes[int(pred)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
            

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(Camera(),mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
