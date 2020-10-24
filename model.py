from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback 
import matplotlib.pyplot as plt 
import numpy as np 


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.97):
            self.model.stop_training = True
callbacks = myCallback()
train_dir = 'TrainData'
val_dir = 'ValData'

train_image_gen = ImageDataGenerator(rescale=1/255, rotation_range=0.2, zoom_range=0.2, shear_range=0.15, horizontal_flip=True)
train_datagen = train_image_gen.flow_from_directory(train_dir, color_mode='grayscale', target_size=(300,400), class_mode='categorical')
val_image_gen = ImageDataGenerator(rescale=1/255)
val_datagen = val_image_gen.flow_from_directory(val_dir, color_mode='grayscale',target_size=(300,400),  class_mode='categorical')
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(300,400,1), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

r = model.fit_generator(train_datagen, validation_data=(val_datagen), epochs=50, callbacks=[callbacks])

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.savefig('graph.png')