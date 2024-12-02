# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Load the digits dataset
digits = datasets.load_digits()

# Print the entire dataset (features) in matrix form (1797 samples with 64 features each)
print('Data:')
print(digits.data)

# Print the labels (target values) corresponding to each digit image
print(f'\nLabels:')
print(digits.target)

# Print the raw pixel data for the first digit image (8x8 array)
print(f'\nDigit image:')
print(digits.images[0])

# Create an instance of the Support Vector Classifier (SVM)
# Parameters:
# - gamma: Controls the influence of a single training example (smaller value = smoother boundary)
# - C: Controls regularization (higher value = less regularization)
clf = svm.SVC(gamma=0.0001, C=100)

# Print the total number of samples in the dataset
print(f'Digits Length: ')
print(len(digits.data))

# Split the dataset into:
# - x: Features (all digit images except the last one)
# - y: Targets/labels (all labels except the last one)
x, y = digits.data[:-1], digits.target[:-1]

# Train the SVM model on the features (x) and labels (y)
clf.fit(x, y)

# Select an image to test the model (index -6 means 6th image from the end)
image_index = -6

# Predict the label for the selected image
# - Reshape the input data to a 2D array (1 sample with 64 features)
prediction = clf.predict(digits.data[image_index].reshape(1, -1))

# Print the model's prediction for the selected digit image
print(f'\nPrediction: {prediction[0]}')

# Display the selected image
# - `cmap=plt.cm.gray_r`: Display the image in grayscale
# - `interpolation="nearest"`: Display the image without smoothing
plt.imshow(digits.images[image_index], cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Prediction: {prediction[0]}")  # Add a title showing the predicted digit
plt.show()


