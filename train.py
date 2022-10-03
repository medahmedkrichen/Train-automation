import os
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import gc
from tqdm import tqdm
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/')
parser.add_argument('--model_name', type=str, default='EfficientNetV2B0')
parser.add_argument('--rescale', action='store_true')
parser.add_argument('--size', type=int, default=256) 
parser.add_argument('--batch_size', type=int, default=64) 
parser.add_argument('--epochs', type=int, default=10) 
opt = parser.parse_args()

# data
    # resolution
    # data
    # batch_size
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        opt.data,
        target_size=(opt.size, opt.size),
        batch_size=opt.batch_size,)


# model
    #N_CLASSES
    
model_dict = {"DenseNet121" : ".densenet.DenseNet121",
"DenseNet169" : ".densenet.DenseNet169",
"DenseNet201" : ".densenet.DenseNet201",

"EfficientNetV2B0" : ".efficientnet_v2.EfficientNetV2B0",
"EfficientNetV2B1" : ".efficientnet_v2.EfficientNetV2B1",
"EfficientNetV2B2" : ".efficientnet_v2.EfficientNetV2B2",
"EfficientNetV2B3" : ".efficientnet_v2.EfficientNetV2B3",
"EfficientNetV2L" : ".efficientnet_v2.EfficientNetV2L",
"EfficientNetV2M" : ".efficientnet_v2.EfficientNetV2M",
"EfficientNetV2S" : ".efficientnet_v2.EfficientNetV2S",

"InceptionResNetV2" : ".inception_resnet_v2.InceptionResNetV2",

"InceptionV3" : ".inception_v3.InceptionV3",

"MobileNet" : ".mobilenet.MobileNet",

"MobileNetV2" : ".mobilenet_v2.MobileNetV2",

"MobileNetV3Large" : ".MobileNetV3Large",

"MobileNetV3Small" : ".MobileNetV3Small",

"NASNetLarge" : ".nasnet.NASNetLarge",

"NASNetMobile" : ".nasnet.NASNetMobile",

"RegNetX002" : ".regnet.RegNetX002",
"RegNetX004" : ".regnet.RegNetX004",
"RegNetX006" : ".regnet.RegNetX006",
"RegNetX008" : ".regnet.RegNetX008",
"RegNetX016" : ".regnet.RegNetX016",
"RegNetX032" : ".regnet.RegNetX032",
"RegNetX040" : ".regnet.RegNetX040",
"RegNetX064" : ".regnet.RegNetX064",
"RegNetX080" : ".regnet.RegNetX080",
"RegNetX120" : ".regnet.RegNetX120",
"RegNetX160" : ".regnet.RegNetX160",
"RegNetX320" : ".regnet.RegNetX320",
"RegNetY002" : ".regnet.RegNetY002",
"RegNetY004" : ".regnet.RegNetY004",
"RegNetY006" : ".regnet.RegNetY006",
"RegNetY008" : ".regnet.RegNetY008",
"RegNetY016" : ".regnet.RegNetY016",
"RegNetY032" : ".regnet.RegNetY032",
"RegNetY040" : ".regnet.RegNetY040",
"RegNetY064" : ".regnet.RegNetY064",
"RegNetY080" : ".regnet.RegNetY080",
"RegNetY120" : ".regnet.RegNetY120",
"RegNetY160" : ".regnet.RegNetY160",
"RegNetY320" : ".regnet.RegNetY320",

"ResNet101" : ".resnet.ResNet101",
"ResNet152" : ".resnet.ResNet152",
"ResNet50" : ".resnet.ResNet50",

"ResNet101V2" : ".resnet_v2.ResNet101V2",
"ResNet152V2" : ".resnet_v2.ResNet152V2",
"ResNet50V2" : ".resnet_v2.ResNet50V2",

"ResNetRS101" : ".resnet_rs.ResNetRS101",
"ResNetRS152" : ".resnet_rs.ResNetRS152",
"ResNetRS200" : ".resnet_rs.ResNetRS200",
"ResNetRS270" : ".resnet_rs.ResNetRS270",
"ResNetRS350" : ".resnet_rs.ResNetRS350",
"ResNetRS420" : ".resnet_rs.ResNetRS420",
"ResNetRS50" : ".resnet_rs.ResNetRS50",

"VGG16" : ".vgg16.VGG16",

"VGG19" : ".vgg19.VGG19",

"Xception" : ".xception.Xception"}


model_name = model_dict[opt.model_name]    
exec(f"base_model = tf.keras.applications{model_name}(input_shape=({opt.size}, {opt.size},3),include_top=False,weights='imagenet')")
#base_model = model_name(input_shape=(256, 256,3),
                                               #include_top=False,
                                               #weights='imagenet')
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
nb_classlen = len(os.listdir(opt.data))
prediction_layer = tf.keras.layers.Dense(nb_classlen,activation='softmax')


inputs = tf.keras.Input(shape=(opt.size, opt.size, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# moodel paremters
    # loss
    # optimizer
    # metric
#loss = tf.keras.losses.BinaryCrossentropy()
model.compile(loss='categorical_crossentropy', 
              optimizer='Adam', 
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

#filepath = "models/saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
#                             save_best_only=False,save_freq='epoch')

#train

history = model.fit(
      train_generator,
      epochs=opt.epochs,
      callbacks=[learning_rate_reduction],
      verbose=1,
      batch_size=opt.batch_size
      )

model.save(f'models/{opt.model_name}.h5')
#tf.keras.models.save_model(model,'model.h5')
"""
best_acc = 0
best_model = ""
for i in os.listdir("models"):
    model.load_weights("models/"+i)
    loss, acc = model.evaluate_generator(test_generator, steps=3, verbose=0)
    if acc > best_acc:
        best_model = i
        best_acc = acc
 """
# simple accuracy
    # delete me

model = load_model(f'models/{opt.model_name}.h5')
loss, acc = model.evaluate(train_generator, steps=3, verbose=0)
acc = acc *100
print(f"Train accuracy is: {acc:.2f}%")