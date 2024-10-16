import xml.etree.ElementTree as xet
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.utils import validation

def preprocessing(df):
    labels = df.iloc[:,1:5].values
    data = []
    output = []
    image_path = df['imagepath']

    for ind in range(len(image_path)):
        image = image_path[ind]
        img_arr = cv2.imread(image)
        h,w,d, = img_arr.shape
        # preprocessing
        load_image = load_img(image, target_size=(224,224))
        load_image_arr = img_to_array(load_image)
        norm_load_image_arr = load_image_arr/255.0
        # normalization to labels
        xmin, xmax, ymin, ymax = labels[ind]
        nxmin, nxmax = xmin/w, xmax/w
        nymin, nymax = ymin/h, ymax/h
        label_norm = (nxmin, nxmax, nymin, nymax)
        data.append(norm_load_image_arr)
        output.append(label_norm)
    
    return data, output

def training(data, output, test_size, learning_rate, batch, epoch):       
    X = np.array(data, dtype=np.float32)
    y = np.array(output, dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    inception_resnet = InceptionResNetV2(weights="imagenet", input_tensor=Input(shape=(224,224,3)))
    inception_resnet.trainable = False
    # ------------------------------
    headmodel = inception_resnet.output
    headmodel = Flatten()(headmodel)
    headmodel = Dense(500, activation='relu')(headmodel)
    headmodel = Dense(250, activation='relu')(headmodel)
    headmodel = Dense(4, activation='sigmoid')(headmodel)

    model = Model(inputs=inception_resnet.input, outputs=headmodel)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    history = model.fit(x=X_train, y=y_train, batch_size=batch, epochs=epoch, validation_data=(X_val, y_val))

    loss_train = model.evaluate(X_train, y_train)
    loss_test = model.evaluate(X_val, y_val)
    
    y_pred_train = model.predict(X_val)
    ious = []
    for pred_bbox, true_bbox in zip(y_pred_train, y_train):
        iou = calculate_iou(pred_bbox, true_bbox)
        ious.append(iou)
    average_iou = sum(ious) / len(ious)
    return model, history, loss_train, loss_test, average_iou

def plot_eval(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return st.pyplot(plt)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2])
    boxBArea = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou