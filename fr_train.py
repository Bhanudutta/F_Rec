import cv2
import numpy as np
import pickle 


print("Enter name of person to train face");

name = input();
pathd="pics/"+name+"/";
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0);
i=1;
k=0;
images=[]
label=[]
print("creating dataset")
while i<41:
    ret,img=cap.read();
    res=0;
    k+=1;
    if k%5 == 0:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray);
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            images.append(roi_gray);
            label.append(i)
            i+=1;
print("training");
(images, label) = [np.array(lis) for lis in [images, label]]	
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images,label);
recognizer.save("recognizers/"+name+"_train.yml");
print("training for",name,"complete");
