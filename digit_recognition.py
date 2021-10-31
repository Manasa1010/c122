import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
print("Hi")
X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())

classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=7500,test_size=2500,random_state=42)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train_scaled,y_train)
y_pred=clf.predict(x_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=grey.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(grey,upper_left,bottom_right,(0,255,0),2)
        #roi-region of interest

        roi=grey[upper_left[1]:bottom_right[1],upper_left[0],bottom_right[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert("L")
        image_resized=image_bw.resize((28,28),Image.ANTIALIAS)
        image_inverted=PIL.ImageOps.invert(image_resized)
        pixel_filter=20
        min_pixel=np.persentile(image_inverted,pixel_filter)
        image_scaled=np.clip(image_inverted-min_pixel,0,255)
        max_pixel=np.max(image_inverted)
        image_scaled=np.asarray(image_scaled)/max_pixel
        test_sample=np.arrey(image_scaled).reshape(1,784)
        test_pred=clf.predict(test_sample)
        print("predicted no is",test_pred)
        cv2.imshow("Suman",grey)
        if(cv2.waitKey(1)& 0*FF==ord("Q")):
            break
    
    except Exception as e:
        pass


cap.telease()
cv2.destroyAllWindows()




