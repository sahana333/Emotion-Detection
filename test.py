import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a simple model to test TensorFlow functionality
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(10,))
])

model.summary()
