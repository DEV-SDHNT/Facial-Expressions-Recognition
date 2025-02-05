import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
#from sklearn.multioutput import MultiOutputClassifier

ds=pd.read_csv('../fer2013.csv')
y=ds['emotion']
x=ds['pixels'].apply(lambda x:np.array(x.split(' '),dtype='int32').reshape(48,48))

#y=ds.iloc[:,:-1]
#x=ds.iloc[:,-1]
#print(type(x))
#print(x.shape)

x=np.stack(x)
classes=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

xtrain,xtest,ytrain,ytest=train_test_split(x.reshape(x.shape[0],-1),y,test_size=0.1,random_state=463)

#model=BernoulliNB()
#model=SVC(kernel="linear",verbose=True)
#model=LogisticRegression(max_iter=400,verbose=True)
#model=DecisionTreeClassifier()
#model=MLPClassifier(hidden_layer_sizes=(150,),max_iter=50,alpha=1e-4,solver='sgd',verbose=10,tol=0.001,random_state=1,learning_rate_init=.001,early_stopping=False)
#model=MultiOutputClassifier(,n_jobs=-1)
model=RandomForestClassifier(n_jobs=-1,verbose=True,n_estimators=200,max_depth=1000)
print("Training :::")

yt=list(ytest)
#print(yt)
model.fit(xtrain,ytrain)
#print(xtest[0].shape)

joblib.dump(model,"fer_model.pkl")

pred=model.predict(xtest)
#print(f"Outputs ::{model.n_outputs_}")
#print(list(pred))
print(f'Model Score ::{model.score(xtrain,ytrain)}')
print(f"Acuu_score::: {accuracy_score(pred,ytest)}")
#    t=int(input("Enter random Testcase :::"))
 #   print(f"Predicted Emotion :: {classes[pred[t]]} ,Actual Emotion :: {classes[yt[t]]}")
#
 #   img=np.array(xtest[t]).reshape(48,48)
  #  plt.figure(0,figsize=(10,16))
   # plt.imshow(img,cmap='gray')
#plt.title(f"Predicted ::{classes[pred[t]]} , Actual :: {classes[yt[t]]}")
#plt.show()
