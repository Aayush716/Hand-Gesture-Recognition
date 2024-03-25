# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
#32 kernels of size (3 * 3) are added 
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
# This code adds the second convolutional layer. Similar to the first convolutional layer, 
# it uses 32 filters of size 3x3, but it doesn't
#  require the input_shape parameter since it follows the previous max-pooling layer.
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# input_shape is going to be the pooled feature maps from the previous convolution layer
#classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
#  This line adds a flatten layer to the model. The flatten layer transforms the 2D feature maps from the 
# previous layers into a 1D vector. This is necessary before connecting these features to fully connected layers for classification.
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
#To avoid overfitting we used Droupout
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()

# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory('data2/train',
#                                                  target_size=(sz, sz),
#                                                  batch_size=10,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = test_datagen.flow_from_directory('data2/test',
#                                             target_size=(sz , sz),
#                                             batch_size=10,
#                                             color_mode='grayscale',
#                                             class_mode='categorical') 
# classifier.fit_generator(
#         training_set,
#         steps_per_epoch=12841, # No of images in training set
#         epochs=5,
#         validation_data=test_set,
#         validation_steps=4268)# No of images in test set


# # Evaluate the model on the test data
# scores = classifier.evaluate_generator(test_set, steps=4268)  # No of images in the test set
# # Print the accuracy
# print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# print('Model Saved')
# classifier.save_weights('model-bw.h5')
# print('Weights saved')

