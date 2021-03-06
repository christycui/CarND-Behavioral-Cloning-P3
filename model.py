import pandas as pd
import numpy as np
from PIL import Image
import pickle
import csv
import os
import cv2
import sklearn
import random

path = '/Users/Christy/Desktop/'
samples = []
with open('driving_log.csv','r') as log_1, open(path+'driving_log.csv','r') as log_2:
  reader_1 = csv.reader(log_1)
  reader_2 = csv.reader(log_2)
  next(reader_1, None)
  for line in reader_1:
    samples.append(line)
  for row in reader_2:
    samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        center_angle = float(batch_sample[3])
        center_image = cv2.cvtColor(cv2.imread(batch_sample[0]),cv2.COLOR_BGR2RGB)
        center_image_T = adjust_brightness(center_image)
        left_image = cv2.cvtColor(cv2.imread(batch_sample[1].strip()),cv2.COLOR_BGR2RGB)
        left_image_T = adjust_brightness(left_image)
        right_image = cv2.cvtColor(cv2.imread(batch_sample[2].strip()),cv2.COLOR_BGR2RGB)
        right_image_T = adjust_brightness(right_image)
        
        correction = 0.25
        steering_left = center_angle + correction
        steering_right = center_angle - correction
        images.extend([center_image, center_image_T, left_image, left_image_T, right_image, right_image_T])
        angles.extend([center_angle, center_angle, steering_left, steering_left, steering_right, steering_right])

      X_train = np.array(images)
      y_train = np.array(angles).reshape([-1,1])
      # add flipped images and angles to avoid bias
      images_flipped = np.fliplr(X_train)
      angles_flipped = np.negative(y_train)
      X_train = np.concatenate((X_train, images_flipped), axis=0)
      y_train = np.concatenate((y_train, angles_flipped), axis=0)
      yield sklearn.utils.shuffle(X_train, y_train)

def adjust_brightness(image):
    rand = random.uniform(0.4, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * rand
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

model = Sequential()
model.add(Cropping2D(cropping=((60,30), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')