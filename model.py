#!/usr/bin/env python
# coding: utf-8

# ## 1. Load trainning data

# In[8]:


import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

# File name
driving_log = "../training-data/driving_log.csv"

# Read data info
rows = []
with open(driving_log) as file:
    reader = csv.reader(file)
    for row in reader:
        rows.append(row)
train_data, valid_data = train_test_split(rows, test_size=0.2)


# ## 2. Create data generator
# 

# In[9]:


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


# In[10]:


# Set batch size
batch_size=128
# generate train data
train_generator = generator(train_data, batch_size=batch_size)
validation_generator = generator(valid_data, batch_size=batch_size)


# ## 3. Design Model and Train with Keras

# In[11]:


from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Conv2D
from math import ceil

# Create model
model = Sequential()
model.add(Lambda(lambda x: x/128.0 - 1.0, input_shape=(160,320,3)))
# If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile model
model.compile(loss="mse", optimizer="adam")
# Train model on train data
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_data)/batch_size), validation_data=validation_generator,validation_steps=ceil(len(valid_data)/batch_size), shuffle=True, epochs=100, verbose=1)

# Save whole model into .h5 file
model.save("model-1-0-2.h5")


# ## 4. Test

# In[15]:


from keras.models import load_model
model_load = load_model("model.h5")

images = []
# Use '/' if data collection ran on linux 
center_image_file = row[0].split('\\')[-1]
img = Image.open("../training-data/IMG/"+center_image_file)
images.append(np.array(img))
img.close()

steering_angle = float(model_load.predict(images[0][None, :, :, :], batch_size=1))
print (steering_angle)


# ## 5. Model Visualization

# In[ ]:


from keras.models import load_model
# Load saved model
model = load_model("model.h5")


# In[2]:


# Visual layers
print(model.summary())


# ## 6. Feature Maps

# In[4]:


from keras import models
# Crop the outputs 7 layers
layer_outputs = [layer.output for layer in model.layers[:7]]
# Creates a new model that will yeild these outputs
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 


# In[6]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
img = Image.open("./examples/center_2019_10_02_02_20_53_427.jpg")
img = np.array(img)
plt.imshow(img)


# In[8]:


# Compute activations
activations = activation_model.predict(img.reshape(1,160,320,3)) # Returns a list of Numpy arrays: one array per layer activation


# In[46]:


# Plot activations
def feature_maps(activation):
    featuremaps = activation.shape[3]
    plt.figure(1, figsize=(20,5))
    for featuremap in range(featuremaps):
        plt.subplot(10,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


# In[50]:


# Output of second layer
normalized_and_croped_layer_activation = activations[1]
feature_maps(normalized_and_croped_layer_activation)


# In[52]:


# Output of first convolutional layer
first_conv_layer_activation = activations[2]
feature_maps(first_conv_layer_activation)


# In[53]:


# Output of second convolutional layer
second_conv_layer_activation = activations[3]
feature_maps(second_conv_layer_activation)

