import cv2
import numpy as np
import pickle 
from matplotlib import pyplot as plt

print("Enter name of person to train face");

name = input();
pathd="pics/"+name+"/";
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0);
i=1;
k=0;
images=[]
label=[]
bl = chr(9608)
print("creating dataset")
print("Please keep facing the webcam and keep your face slightly moving to train if from all directions");
while i<41:
    ret,img=cap.read();
    res=0;
    k+=1;
    if k%5 == 0:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray);
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            images.append(roi_gray);
            label.append(i)
            i+=1;
            perc = 100*(i/41)
            perc = int(perc)
            print(bl*(perc//4),str(perc)+"%",end="\r");
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    #plt.pause(0.01)
print("training");
(images, label) = [np.array(lis) for lis in [images, label]]	
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images,label);
recognizer.save("recognizers/"+name+"_train.yml");
print("training for",name,"complete");
