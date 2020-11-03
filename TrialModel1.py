import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('D:/tacti and tact/images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 150,
                                                 class_mode='binary',
                                                 shuffle= True)
                                                
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_datagen.flow_from_directory('D:/tacti and tact/images/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn = tf.keras.models.Sequential([Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
                                  MaxPool2D(pool_size=2, strides=2),

                                  Flatten(),
                                  Dense(units=64, activation='relu'),
                                  Dense(units=1, activation='sigmoid')
                          ])                                        
                    
                                  
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = validation_set, epochs =10 )

test_image = image.load_img('D:/tacti and tact/Plastic-Yogurt-Pot-Foil-Lids.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'cut'
else:
  prediction = 'uncut'

print(prediction)  




