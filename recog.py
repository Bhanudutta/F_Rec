import cv2
import numpy as np
import pickle 
from os import walk
from matplotlib import pyplot as plt

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
images=[];
path=""


def filelist(path):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    return f;
    

flist = filelist('recognizers')
name = [x[:-10] for x in flist]
print(name)
recogs = [];
for n in name:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizers/"+n+"_train.yml");
    recogs.append(recognizer)
cap=cv2.VideoCapture(0);
i=0;


while(1):
    ret,img=cap.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        confc = [0]*len(name);
        j = 0;
        for recognizer in recogs:
            nbr_predicted, conf = recognizer.predict(roi_gray)
            confc[j] = conf;
            j+=1;
        if min(confc)<120:
            mini = confc.index(min(confc));
            cv2.putText(img,name[mini],(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        else:
            cv2.putText(img,'Unknown',(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #out.write(img);	
    #cv2.imshow("out",img);
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.pause(0.01)
    #cv2.imwrite("dta/"+str(i)+".jpg",img)
    cv2.waitKey(1);
    i+=1;
cv2.destroyAllWindows()
