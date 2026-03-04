import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='isl_balanced_2.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with one of your training images
img = Image.open('dataset\A\A_1768210879646.jpg')  # Use YOUR actual path
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_batch)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

top_idx = predictions.argmax()
print(f"Image should be 'A'")
print(f"Model predicts: {labels[top_idx]} ({predictions[top_idx]*100:.1f}%)")
print(f"Top 3: {labels[predictions.argsort()[-3:][::-1]]}")