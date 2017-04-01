import pandas as pd
import numpy as np
from PIL import Image
import pickle
import csv
import os
import cv2
import sklearn
import random

path = "../../thunderhill/udacity-day-01-exported-1102/"

samples = []
with open(path+'output.txt','r') as log_1:
  reader_1 = csv.reader(log_1)
  #reader_2 = csv.reader(log_2)
  #next(reader_1, None)
  #next(reader_2, None)
  for line in reader_1:
    samples.append(line)
  #for row in reader_2:
  #  samples.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      output = []
      for batch_sample in batch_samples:
        center_angle = float(batch_sample[11])
        if center_angle == 0:
          continue
        throttle = float(batch_sample[12])
        brake = float(batch_sample[13])
        if brake > 0:
          throttle = -brake
        filepath = open(path+batch_sample[0],'rb')
        img = np.array(Image.frombytes('RGB',[960,480],filepath.read(),'raw'))
        img = cv2.resize(img, (160, 320))
        center_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        center_image_T = adjust_brightness(center_image)

        
        single_output = [center_angle,throttle]
        images.extend([center_image, center_image_T])
        output.extend([single_output, single_output])

      X_train = np.array(images)
      y_train = np.array(output).reshape([-1,2])
      # add flipped images and angles to avoid bias
      images_flipped = np.fliplr(X_train)
      angles_flipped = np.array([[-angle,throttle] for [angle,throttle] in output]).reshape([-1,2])
      X_train = np.concatenate((X_train, images_flipped), axis=0)
      y_train = np.concatenate((y_train, angles_flipped), axis=0)
      yield sklearn.utils.shuffle(X_train, y_train)""

def adjust_brightness(image):
    rand = random.uniform(0.4, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * rand
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, Cropping2D, Dropout, merge
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
model.add(Dense(2))

# image_input = Input(shape=(160,320,3), dtype='int32', name='image_input')
# x = Cropping2D(cropping=((60,30), (0,0)))(image_input)
# x = Lambda(lambda x: (x / 255.0) - 0.5)(x)
# x = Convolution2D(24, 5, 5, subsample=(2,2), activation='relu')(x)
# x = Convolution2D(36, 5, 5, subsample=(2,2), activation='relu')(x)
# x = Convolution2D(48, 5, 5, subsample=(2,2), activation='relu')(x)
# x = Convolution2D(64, 3, 3, activation='relu')(x)
# x = Convolution2D(64, 3, 3, activation='relu')(x)
# steering_output = Flatten(1)(x)

# speed_input = Input(shape=(1,), name='speed_input')
# x = keras.layers.concatenate([image_flattened, speed_input])

# x = Dense(1, activation='relu')(x)
# model = Model(inputs=[image_input, speed_input], outputs=[steering_output, throttle_output])

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('thunderhill.h5')