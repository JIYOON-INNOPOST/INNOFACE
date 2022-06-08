import torch
import tensorflow as tf
import math
import numpy as np
import cv2
from tensorflow import keras



def NME(y_true,y_pred):
    w = 256
    h = 256
    d = (w**2+h**2)**(1/2)
    
    nose_x = y_true[:,0]
    nose_x_pred = y_pred[:,0]
    nose_y = y_true[:,1]
    nose_y_pred = y_pred[:,1]
    distance_nose = (tf.math.sqrt((nose_x_pred - nose_x)**2 + (nose_y_pred - nose_y)**2))/d
    left_eye_x = y_true[:,2]
    left_eye_x_pred = y_pred[:,2]
    left_eye_y = y_true[:,3]
    left_eye_y_pred = y_pred[:,3]
    distance_left_eye = (tf.math.sqrt((left_eye_x_pred - left_eye_x)**2 + (left_eye_y_pred - left_eye_y)**2))/d
    right_eye_x = y_true[:,4]
    right_eye_x_pred = y_pred[:,4]
    right_eye_y = y_true[:,5]
    right_eye_y_pred = y_pred[:,5]
    distance_right_eye = (tf.math.sqrt((right_eye_x_pred - right_eye_x)**2 + (right_eye_y_pred - right_eye_y)**2))/d
    left_mouse_x = y_true[:,6]
    left_mouse_x_pred = y_pred[:,6]
    left_mouse_y = y_true[:,7]
    left_mouse_y_pred = y_pred[:,7]
    distance_left_mouse = (tf.math.sqrt((left_mouse_x_pred - left_mouse_x)**2 + (left_mouse_y_pred - left_mouse_y)**2))/d
    right_mouse_x = y_true[:,8]
    right_mouse_x_pred = y_pred[:,8]
    right_mouse_y = y_true[:,9]
    right_mouse_y_pred = y_pred[:,9]
    distance_right_mouse = (tf.math.sqrt((right_mouse_x_pred - right_mouse_x)**2 + (right_mouse_y_pred - right_mouse_y)**2))/d
    left_ear_x = y_true[:,10]
    left_ear_x_pred = y_pred[:,10]
    left_ear_y = y_true[:,11]
    left_ear_y_pred = y_pred[:,11]
    distance_left_ear = (tf.math.sqrt((left_ear_x_pred - left_ear_x)**2 + (left_ear_y_pred - left_ear_y)**2))/d
    right_ear_x = y_true[:,12]
    right_ear_x_pred = y_pred[:,12]
    right_ear_y = y_true[:,13]
    right_ear_y_pred = y_pred[:,13]
    distance_right_ear = (tf.math.sqrt((right_ear_x_pred - right_ear_x)**2 + (right_ear_y_pred - right_ear_y)**2))/d
    
    nme = (tf.reduce_mean(distance_right_ear) + tf.reduce_mean(distance_left_ear) +  tf.reduce_mean(distance_right_mouse) + tf.reduce_mean(distance_left_mouse) + tf.reduce_mean(distance_right_eye) + tf.reduce_mean(distance_left_eye) + tf.reduce_mean(distance_nose))

    return nme



class Image_Crop:
    def __init__(self, yolo_path, model_path):
        self.yolo = torch.load(yolo_path)
        self.model = keras.models.load_model(model_path, custom_objects = {'NME' : NME})
    
    
    def predict_landmarks(self, image):
        
        img_size = 256
        
        
        try:
            result = self.yolo(image , augment=True)
            crop_result = result.crop()
            
            x,y,w,h, = int(crop_result[0]['box'][0]), int(crop_result[0]['box'][1]), int(crop_result[0]['box'][2]), int(crop_result[0]['box'][3])

            crop_img = image[y:y+(h-y), x:x+(w-x)]
            
            crop_size = crop_img.shape
            
            # Preprocessing for Prediction
            image = cv2.resize(crop_img, (img_size, img_size))
            image = image.reshape((1,img_size,img_size,3))
            image = image/255.
            predict_landmarks = self.model.predict(image)[0]
            
            dis_x, dis_y = crop_size[1] ,crop_size[0]
            for i in range(0,14,2):
                cv2.circle(crop_img, (int(predict_landmarks[i]*(dis_x/256)),int(predict_landmarks[i+1] *(dis_y/256))), 5, (0,255,0), 2, cv2.LINE_AA)

            return crop_img
        
        except:
            return image
    
    
    
    def for_cam(self, image):
        org_img = image
        img_size = 256
        result = self.yolo(image , augment=True)
        
        try:
            crop_result = result.crop()
            
            x,y,w,h, = int(crop_result[-1]['box'][0]), int(crop_result[-1]['box'][1]), int(crop_result[-1]['box'][2]), int(crop_result[-1]['box'][3])
            
            crop_img = image[y:h, x:w]
            
            crop_size = crop_img.shape
            
            # Preprocessing for Prediction
            image = cv2.resize(crop_img, (img_size, img_size))
            image = image.reshape((1,img_size,img_size,3))
            image = image/255.
            predict_landmarks = self.model.predict(image)[0]
            
            for i in range(0,14,2):
                tx = int((predict_landmarks[i])*(crop_size[1]/256) + x)
                ty = int((predict_landmarks[i+1])*(crop_size[0]/256) + y)
                cv2.circle(org_img, (tx,ty), 5, (0,255,0), 2, cv2.LINE_AA)
                
            return org_img
        
        except:
            return image
        
        