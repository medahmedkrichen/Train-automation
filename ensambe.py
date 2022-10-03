import os
import cv2
import xgboost as xgb
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


full_pred = []
full_results = []
for model_name in os.listdir("models/"):
    preds = []
    results = []
    model = load_model("models/"+model_name)
    classs = os.listdir("data/")
    classs.sort()
    for clas in range(len(classs)):
        classes = os.listdir("data/")[clas] 
        for i in os.listdir(f"data/{classes}"):
            img = cv2.imread(f"data/{classes}/{i}")
            if img is None :
                pass
            else: 
                img.resize(1,256,256,3)
                pred = model(img)
                preds.append(pred.numpy().argmax())
                results.append(clas)
            
        
    full_pred.append(np.array(preds))
    #results.append(results)
    
full_pred = np.array(full_pred).T
results = np.array(results)

X_train, X_test, y_train, y_test = train_test_split(full_pred, results, test_size=0.1)
print(full_pred)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
acc = accuracy_score(y_pred,y_test)

print(f"Ensemble accuracy is: {acc:.2f}%")