# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/modelarchitecture.png "Model Visualization"
[image2]: ./examples/center-image.jpg "Center Camera"
[image3]: ./examples/roi.png "Croping and Normalization"
[image4]: ./examples/firstconv.png "First Convolutional Layer"
[image5]: ./examples/secondconv.png "Second Convolutional Layer"

---
### Files Structure:

#### 1. Files structure:

The project includes:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Run the code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
---
### Model Architecture and Training Strategy

#### 1. Model architecture design

At first glance, the problem is to find the steering angle and the speed of the car with input from image(s) that capture from three cameras mounted on board. This is a regression problem. To create feature map from input image(s), several convolutional layer can be used. The model should also includes multiple dense layers to handle nonlinearity.
Due to the constrain of the data format that the simulatior sends and receives, all input preprocessing steps has to be included inside the model.

#### 2. Reduce overfitting

To reduce overfitting, driving-clockwise data also was collected eventhough during the autonomous mode, the simulator only drive counter clock-wise direction.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data
Training data was chosen to keep the vehicle driving in the middle the road. Also, recovering from left and right of the road was also collected by placing the car off the center and start recording while driving back to the center.
Trainning data from center camera is sufficient enough for the car to perform well on the test track, so no further data augmentation was performed.

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a neuron net with serveral dense layers. However, the result was not very good and the car seem too sensitive to the changing of input images.

Few convolutional layers was used to filter out noise and to create feature maps for the model. Croping and normalization also was used to help the model train faster and perform better.


#### 2. Final Model Architecture

The final model that is used in this project was deverloped by NVIDIA autonomous driving team. More information cound be found  [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

![alt text][image1]

Here is a sumarry of the architecture:

```python
from keras.models import load_model
model = load_model("model.h5")
print (model.summary())
```
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lambda_2 (Lambda)            (None, 160, 320, 3)       0         
    _________________________________________________________________
    cropping2d_2 (Cropping2D)    (None, 65, 320, 3)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 31, 158, 24)       1824      
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 14, 77, 36)        21636     
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 5, 37, 48)         43248     
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 3, 35, 64)         27712     
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 1, 33, 64)         36928     
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 2112)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 100)               211300    
    _________________________________________________________________
    dense_6 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                510       
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 348,219
    Trainable params: 348,219
    Non-trainable params: 0
    _________________________________________________________________
    None


#### 3. Creation of the Training Data & Training Process

To capture good driving behavior, serveral laps around the track was recorded. Here is anexample of image captured by center camera:

![alt text][image2]

Data generator was created due to a tremendous amount of data was collected and could not be stored onto memory:
```python
# Define genorator for training
from random import shuffle
def generator(data, batch_size = 128):
    data_size = len(data)
    # Loop for ever in generator loop
    while 1:
        shuffle(data)
        for offset in range(0, data_size, batch_size):
            batch = data[offset:offset+batch_size]
            images = []
            mesurements = []
            for x in batch:
                # This is a work around because train data was collected on both Windows and Linux
                # Hence path file contain different deliminator ('/' and '\')
                center_image_file = x[0].split('center_')[-1]
                img = Image.open("../training-data/IMG/center_"+center_image_file)
                images.append(np.array(img))
                img.close()
                mesurements.append(float(x[3]))
                
            # Convert into np array
            x_train = np.array(images)
            y_train = np.array(mesurements)
            
            yield x_train, y_train
```

After data preparation, model was trained and saved into .h5 file:

```python
# Compile model
model.compile(loss="mse", optimizer="adam")
# Train model on train data
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_data)/batch_size), validation_data=validation_generator,validation_steps=ceil(len(valid_data)/batch_size), shuffle=True, epochs=100, verbose=1)

# Save whole model into .h5 file
model.save("model.h5")
```
#### 4. Model visualization of internal model:
To see how model learn from training data, we plot feature maps that were created by inner layers of model:

Croping and normalization:
![alt text][image3]

First convolutional layer:
![alt text][image4]

Second convolutional layer:
![alt text][image5]

We can see that model picks up the lane marks and the detail of the side of the road and learn how to react to these information.
