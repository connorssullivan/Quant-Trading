import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

#Print the didgets data
print('Data:')
print(digits.data)

#Print the label
print(f'\nLables:')
print(digits.target)

# Prit the image
print(f'\nDigit image:')
print(digits.images[0])

# Create an instance of Support Vector Classification (The bondaries for different objects)
clf = svm.SVC(gamma= 0.0001, C=100)

print(f'Digits Length: ')
print(len(digits.data))

#Seperate ddata and the target
x,y = digits.data[:-1], digits.target[:-1]

# Fit the data
clf.fit(x,y)

image_index = -6

# Predict the data
prediction = clf.predict(digits.data[image_index].reshape(1, -1))  # Reshape to 2D array
print(f'\nPrediction: {prediction[0]}')

# Show the image
plt.imshow(digits.images[image_index], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()




