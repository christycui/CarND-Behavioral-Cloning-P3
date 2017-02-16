import pandas as pd
import numpy as np
from PIL import Image
import pickle
import csv
import os
import cv2
import sklearn

path = '../../../../Desktop/'
samples = []
with open(path+'driving_log.csv','r') as log:
  reader = csv.reader(log)
  next(reader, None)
  for line in reader:
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
                center_image = cv2.imread(path+batch_sample[0])
                left_image = cv2.imread(path+batch_sample[1].strip())
                right_image = cv2.imread(path+batch_sample[2].strip())
                center_angle = float(batch_sample[3])
                correction = 0.18
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                images.extend([center_image, left_image, right_image])
                angles.append([center_angle, steering_left, steering_right])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles).reshape([-1,1])
            # add flipped images and angles to avoid bias
            images_flipped = np.fliplr(X_train)
            angles_flipped = -y_train
            X_train = np.concatenate((X_train, images_flipped), axis=0)
            y_train = np.concatenate((y_train, angles_flipped), axis=0)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

model = Sequential()
model.add(Cropping2D(cropping=((60,30), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D((5, 5)))
model.add(Dropout(0.75))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=20)

model.save('model.h5')