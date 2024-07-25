# ~*~ coding: utf-8 ~*~

from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np

# Load our model
model = load_model("model.h5")

# Set our labels from trained data
classes = ["bird", "plane"]

# Load a sample image and resize it
image = load_img("/home/jesse/Downloads/ml_demo/p3.jpg", target_size=(384, 384))

# Generate a NumPy array from image data (input)
img = np.array(image)

# Transformations
img = img / 255.0
img = img.reshape(1, 384, 384, 3)

# Run prediction on image
prediction = model.predict(img)
mpos = np.argmax(prediction)
prediction_label = sorted(classes)[mpos]

# Output prediction from values
print(prediction_label)
