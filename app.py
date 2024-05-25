# # import system libs
# import os
# import time
# import shutil
# import pathlib
# import itertools
# from PIL import Image

# # import data handling tools
# import cv2
# import numpy as np
# import pandas as pd
# import seaborn as sns
# sns.set_style('darkgrid')
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# # import Deep learning Libraries
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam, Adamax
# from tensorflow.keras.metrics import categorical_crossentropy
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
# from tensorflow.keras import regularizers

# # Ignore Warnings
# import warnings
# warnings.filterwarnings("ignore")

# print ('modules loaded')

# import tensorflow as tf
# from flask import Flask, request, jsonify
# from PIL import Image
# import numpy as np
# import io

# app = Flask(__name__)

# # Load your pre-trained model
# loaded_model = tf.keras.models.load_model('model.h5', compile=False)
# loaded_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
#                      loss='categorical_crossentropy', metrics=['accuracy'])
# classes = ['adidas', 'converse', 'nike']

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     # Preprocess the image
#     image = Image.open(io.BytesIO(file.read()))
#     img = image.resize((224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Make prediction
#     predictions = loaded_model.predict(img_array)
#     score = tf.nn.softmax(predictions[0]).numpy()
#     predicted_class = classes[np.argmax(score)]
    
#     return jsonify({'predicted_class': predicted_class})

# if __name__ == '__main__':
#     app.run(debug=True)





# ----------------------------------------------------


# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load your pre-trained model
# loaded_model = tf.keras.models.load_model('model.h5', compile=False)
# loaded_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
#                      loss='categorical_crossentropy',
#                      metrics=['accuracy'])

# classes = ['adidas', 'converse', 'nike']

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # Preprocess the image
#     image = Image.open(io.BytesIO(file.read()))
#     img = image.resize((224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, axis=0)

#     # Make prediction
#     predictions = loaded_model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     predicted_class = classes[tf.argmax(score)]

#     return jsonify({'predicted_class': predicted_class})

# if __name__ == '__main__':
#     app.run(debug=True)

# --------------------------------------------

# import tensorflow as tf
# from flask import Flask, request, jsonify
# from PIL import Image
# import numpy as np
# import io
# from tensorflow.keras.utils import custom_object_scope

# app = Flask(__name__)

# # Load your pre-trained model with custom objects
# with custom_object_scope({'TFOpLambda': tf.keras.layers.Lambda}):
#     loaded_model = tf.keras.models.load_model('model.h5', compile=False)
# loaded_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
#                      loss='categorical_crossentropy', metrics=['accuracy'])
# classes = ['adidas', 'converse', 'nike']

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     # Preprocess the image
#     image = Image.open(io.BytesIO(file.read()))
#     img = image.resize((224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Make prediction
#     predictions = loaded_model.predict(img_array)
#     score = tf.nn.softmax(predictions[0]).numpy()
#     predicted_class = classes[np.argmax(score)]
    
#     return jsonify({'predicted_class': predicted_class})

# if __name__ == '__main__':
#     app.run(debug=True)


# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# import numpy as np

# # Load the pre-trained model
# model = keras.models.load_model('model.h5')

# # Define the classes
# classes = ['adidas', 'converse', 'nike']  # Replace with your actual class names

# # Load and preprocess the image
# image_path = '/path/to/your/image.jpg'  # Replace with the path to your image
# image = Image.open(image_path)
# image = image.resize((224, 224))
# img_array = keras.preprocessing.image.img_to_array(image)
# img_array = np.expand_dims(img_array, axis=0)

# # Make predictions
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# predicted_class = classes[np.argmax(score)]

# print(f"Predicted class: {predicted_class}")



# ------------------------

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')
from tensorflow import keras
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# Ensure you have updated TensorFlow and Keras to 2.9.0
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Define the custom function if needed
def custom_activation(x):
    return K.relu(x)  # Replace with your custom logic if needed

# Load the pre-trained model with custom objects (if any)
# custom_objects = {
#     'TFOpLambda': Lambda,
#     'custom_activation': custom_activation,  # Add your custom function if used
# }

# Load the model
model_path = 'model.h5'  # Ensure the correct path to the downloaded model
# model = keras.models.load_model(model_path, custom_objects=custom_objects)
model = keras.models.load_model(model_path)

# Define the classes
classes = ['adidas', 'converse', 'nike']  # Replace with your actual class names

# Load and preprocess the image
image_path = 'testimage2.jpg'  # Replace with the path to your image
image = Image.open(image_path)
image = image.resize((224, 224))
img_array = keras.preprocessing.image.img_to_array(image)
img_array = np.expand_dims(img_array, axis=0)

# Make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = classes[np.argmax(score)]

print(f"Predicted class: {predicted_class}")
