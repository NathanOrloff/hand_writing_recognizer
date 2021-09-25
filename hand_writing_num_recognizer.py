import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize

print(1)
model = tf.keras.models.load_model("my_model")
print(2)
image = np.array(imread("number.png")/255.0, dtype='float32')
print(3)


print(4)

image = np.array(image)[:,:,:,np.newaxis]
print(5)
test_data = tf.convert_to_tensor(image)
print(6)
probabilities = model.predict(test_data)

print(probabilities)

