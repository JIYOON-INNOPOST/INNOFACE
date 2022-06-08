import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization
import torch
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import crop

if __name__=="__main__":
    tempy = crop.Image_Crop(yolo_path='./data/epoch_4_facedetectionmodel.pt', model_path='./data/stratify_dataset_light_cnn.h5')
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=default)
    args = parser.parse_args()
    img = plt.imread(args.img_path[0])
    tempy1 = tempy.for_cam(img)
    Image.fromarray(tempy1)