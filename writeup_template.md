#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 a video of the car driving itself around the track completing the full loop

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a cropping layer, a normalization layer, three convolutional layers with 2x2 strides and 5x5 filters, two additional non-strided convolutional layers with 3x3 filters and finally a fully connected layer. (model.py lines 76-84) 

The model includes RELU layers to introduce nonlinearity (code line 78 to 82), and the data is normalized in the model using a Keras lambda layer (code line 77). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Compared to the NVIDIA model, this model has fewer fully connected layers because of smaller size of training data, in order to prevent overfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

I played around with the batch size for the generator, and found that batch size equals 16 runs faster while maintaining the model quality.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center images as well as left and right images with steering angle adjustments.

I recorded more data for driving between the bridge and after the dirt road after the bridge, because those are tricker road conditions.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build from a simple network.

I started with one non-strided convolutional layer after cropping and normalization. I improved by increasing the size of cropping, keeping only the actual roads and excluding scenearies. 

Then I gradually increased the number of non-strided layers, but soon I realize that not only was it taking up too much computer power, it was also inefficient and the model did not perform well. 

So I moved towards the NVIDIA model by starting with 2x2 strided layers, followed by non-strided layers. My model improved but the car was not able to recognize a dirt road.

Because of that, I introduced brightness adjustment. I converted the images into HSV and changed the 'V' value of the images to adjust brightness. The model improved and started turning in the right direction when there was no lane marking, but it was clear that the steering was not enough to make it back to the middle of the road.

I played with the steering adjustment angle. But it didn't help much. 

I was stuck at this point for a long time... until I decided to re-read the NVIDIA paper thoroughly to see if I missed anything. I had previously skimmed through it but this time I took detailed notes.

The final adjustment I made to the model is removing frames with 0 steering angle. I got the idea from the data collection section of the NVIDIA paper. Specially, "to remove a bias towards driving straight the training data includes a higher proportion of frames that represent road curves." I realized that my model wasn't working not because it was not complicated enough, but because my data was biased towards driving straight, and therefore was not able to make a turn as big as it needed to recover back to the middle of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| Layer             |     Description                   | 
|:---------------------:|:---------------------------------------------:| 
| Input             | 320x160x3 RGB image                 | 
| Cropping          | Keeping only road portion of the image                 | 
| Normalization     | normalize pixel values to have a zero mean                | 
| Convolution 5x5 with RELU      | 2x2 stride, valid padding, output depth of 24  |
| Convolution 5x5 with RELU      | 2x2 stride, valid padding, output depth of 36  |
| Convolution 5x5 with RELU      | 2x2 stride, valid padding, output depth of 48  |                     |
| Convolution 3x3 with RELU      | non-strided, output depth of 64 |
| Convolution 3x3 with RELU      | non-strided, output depth of 64 |
| Flatten       |
| Fully connected   | outputs 1                          |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the Udacity-provided driving dataset, because my keyboard driving was not ideal after several attempts.

I then recorded the portion of the road after the bridge where there is no clear lane markings, in hopes of gathering more data for tricker road conditions.

I augmented the data by using both left and right cameras and adjusting the steering angle accordingly for those images. I further augmented the data by making copies of the images with different brightness.

I also flipped images and angles, doubling the size of the dataset.

After the collection process, I actually cut down the images with a steering angle of zero, in order to balance the dataset so that the model is not biased towards driving straight.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the decline in both training and validation MSE. I used an adam optimizer so that manually training the learning rate wasn't necessary.
