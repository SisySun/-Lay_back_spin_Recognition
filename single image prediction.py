from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


def preview_image(img_path):
    # Load the image from file
    img = image.load_img(img_path, target_size=(224, 224))

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Load the saved model
model = load_model('figure_skating_spin_classifier.h5')
# Load an image from file
img_path = 'single images/image3.jpg'
preview_image(img_path)
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand dimensions to match the model's expected input shape
img_array = np.expand_dims(img_array, axis=0)

# Normalize the image (optional, depends on how your model was trained)
img_array = img_array / 255.0

# Optionally, you can preprocess the image further according to your model's requirements (e.g., scaling, centering)
# Predict the class probabilities (output will be probabilities for class 1)
predictions = model.predict(img_array)

# Since it's a binary classification, you might want to interpret the output
if predictions[0] > 0.5:
    print("Figure skating layback spin")
else:
    print("Not a figure skating layback spin")
