import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model('model.h5')

print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: We can apply optimization to shrink it even further!
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print("Saving TFLite model...")
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Done! model.tflite created successfully.")
