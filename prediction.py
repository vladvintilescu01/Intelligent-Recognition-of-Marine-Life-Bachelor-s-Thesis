# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:32:30 2024

@author: vladv
"""

#prediction on VGG

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import tensorflow as tf

# import numpy as np
# from keras.preprocessing import image
# from keras.models import model_from_json
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# from keras.optimizers import Adam, RMSprop

# # later...
# # load json and create model
# path = "D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\"
# json_file = open(path+'Fish_DataSet_VGG.json', 'r')
# loaded_model_json = json_file.read()
# ##
# json_file.close()
# model = model_from_json(loaded_model_json)
# ### load weights into new model
# model.load_weights("Fish_DataSet_VGG.weights.h5")
# opt = Adam(learning_rate=0.0001)
# model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# #new prediction
# #Prediction on an image

# CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream"])
# subdir = 'D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\test\\001.png'

# height =224
# width = 224
# image = cv2.imread(subdir)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(Image.fromarray(image))
# plt.show()
# image = cv2.resize(image, (height, width))
# image_array = np.asarray(image)
# image_array = image_array / 255.0  # Normalize the pixel values between 0 and 1
# print(image_array.shape)
# # Reshape the image array to match the input shape of your model
# image_array = image_array.reshape(1, width, height, 3)
# prediction = model.predict(image_array)
# class_prediction =CLASSES[np.argmax(prediction, axis = 1)]
# print(class_prediction)

# #prediction on Inception

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import tensorflow as tf

# import numpy as np
# from keras.preprocessing import image
# from keras.models import model_from_json
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# from keras.optimizers import Adam, RMSprop

# # later...
# # load json and create model
# path = "D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\"
# json_file = open(path+'Fish_DataSet_Inception.json', 'r')
# loaded_model_json = json_file.read()
# ##
# json_file.close()
# model = model_from_json(loaded_model_json)
# ### load weights into new model
# model.load_weights("Fish_DataSet_Inception.weights.h5")
# opt = Adam(learning_rate=0.0001)
# model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# #new prediction
# #Prediction on an image

# CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream"])
# subdir = 'D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\test\\002.png'

# height =224
# width = 224
# image = cv2.imread(subdir)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(Image.fromarray(image))
# plt.show()
# image = cv2.resize(image, (height, width))
# image_array = np.asarray(image)
# image_array = image_array / 255.0  # Normalize the pixel values between 0 and 1
# print(image_array.shape)
# # Reshape the image array to match the input shape of your model
# image_array = image_array.reshape(1, width, height, 3)
# prediction = model.predict(image_array)
# class_prediction =CLASSES[np.argmax(prediction, axis = 1)]
# print(class_prediction)


#prediction on ResNet50

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import tensorflow as tf

# import numpy as np
# from keras.preprocessing import image
# from keras.models import model_from_json
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# from keras.optimizers import Adam, RMSprop

# # later...
# # load json and create model
# path = "D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\"
# json_file = open(path+'Fish_DataSet_ResNet50.json', 'r')
# loaded_model_json = json_file.read()
# ##
# json_file.close()
# model = model_from_json(loaded_model_json)
# ### load weights into new model
# model.load_weights("Fish_DataSet_ResNet50.weights.h5")
# opt = Adam(learning_rate=0.0001)
# model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# #new prediction
# #Prediction on an image

# CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream"])
# subdir = 'D:\\Facultate_ACE\\Facultate_Anul_IV\\ML\\test\\002.png'

# height =224
# width = 224
# image = cv2.imread(subdir)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(Image.fromarray(image))
# plt.show()
# image = cv2.resize(image, (height, width))
# image_array = np.asarray(image)
# image_array = image_array / 255.0  # Normalize the pixel values between 0 and 1
# print(image_array.shape)
# # Reshape the image array to match the input shape of your model
# image_array = image_array.reshape(1, width, height, 3)
# prediction = model.predict(image_array)
# class_prediction =CLASSES[np.argmax(prediction, axis = 1)]
# print(class_prediction)


#prediction on ResNet50-optimized

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import tensorflow as tf
# import numpy as np
# import cv2
# from keras.models import model_from_json
# from keras.optimizers import Adam
# import matplotlib.pyplot as plt
# from PIL import Image

# # Load model architecture and weights
# path = "D:/Facultate_ACE/Facultate_Anul_IV/ML/"
# with open(path + 'Fish_DataSet_ResNet50.json', 'r') as json_file:
#     loaded_model_json = json_file.read()

# model = model_from_json(loaded_model_json)
# model.load_weights(path + "Fish_DataSet_ResNet50.weights.h5")

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Classes — update if possible from train_gen.class_indices in training phase
# CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet"])  # all types of fish
# # Load and preprocess image
# image_path = "D:/Facultate_ACE/Facultate_Anul_IV/ML/test/00004.png" 
# img = cv2.imread(image_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.axis("off")
# plt.show()

# img = cv2.resize(img, (224, 224))
# img = img.astype('float32') / 255.0
# img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)

# # Predict
# pred = model.predict(img)
# predicted_class = CLASSES[np.argmax(pred)]
# print("Predicted class:", predicted_class)


#prediction on DenseNet121-optimized

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
import numpy as np
import cv2
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image

# Load model architecture and weights
path = "D:/Facultate_ACE/Facultate_Anul_IV/ML/"
with open(path + 'Fish_DataSet_DenseNet121_v2.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights(path + "Fish_DataSet_DenseNet121_v2.weights.h5")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Classes — update if possible from train_gen.class_indices in training phase
#CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet"])  # all types of fish
CLASSES = np.array(["Catfish", "Glass Perchlet", "Goby", "Gourami", "Grass Carp", "Knifefish", "Silver Barb", "Tilapia"])  # all types of fish
# Load and preprocess image
image_path = "D:/Facultate_ACE/Facultate_Anul_IV/ML/FishImgDataset/test/Catfish/21a6d418-c380-4676-98fc-edb8aa65a4d3-0mm.jpg" 
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()

img = cv2.resize(img, (224, 224))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)

# Predict
pred = model.predict(img)
predicted_class = CLASSES[np.argmax(pred)]
print("Predicted class:", predicted_class)

