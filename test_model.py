import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--testdir', type=str, default='data/')
parser.add_argument('--model', type=str, default='model.h5')
parser.add_argument('--resize', type=int, default=256)  
parser.add_argument('--greyscale', action='store_true')  
parser.add_argument('--rescale',action='store_true')
opt = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="-1"    


model = load_model(opt.model)

if opt.rescale == True:
    test_datagen = ImageDataGenerator(rescale=1/255)
else:
    test_datagen = ImageDataGenerator()
    
length = opt.resize


if opt.greyscale == True:
    train_set = test_datagen.flow_from_directory(directory=opt.testdir,target_size=(length, length),color_mode="grayscale")
else: 
    train_set = test_datagen.flow_from_directory(directory=opt.testdir,target_size=(length, length),color_mode="rgb")


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val    

def get_recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return recall    

def get_precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return precision   

model.compile(loss='categorical_crossentropy',metrics=[get_f1,'accuracy',get_recall,get_precision])
results = model.evaluate(train_set,return_dict=True,verbose=0)
print(results)