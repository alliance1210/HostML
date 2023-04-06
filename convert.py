import tensorflow as tf
from tensorflow import keras

# Load Keras model
model = keras.models.load_model('BestModel_v3.h5')

# Convert Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TensorFlow Lite model to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)