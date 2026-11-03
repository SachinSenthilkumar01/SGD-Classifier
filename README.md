# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start the program.
2. Import the required libraries NumPy, Pandas, Matplotlib and SGDClassifier.
3. Load the Iris dataset containing sepal length, sepal width, petal length and petal width.
4. Split the dataset into training and testing data.
5. Create an SGD Classifier model with suitable parameters.
6. Train the model using the fit() function.
7. Predict the Iris flower species using the predict() function.
8. Calculate and display the accuracy of the model.
9. Get the flower measurements from the user.
10. Predict and display the species of the Iris flower.
11. Stop the program.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SACHIN S.B
RegisterNumber:  212225040353
*/
```
```
# Program to predict Iris flower species using SGD Classifier
# Developed by:
# RegisterNumber:

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Classifier
model = SGDClassifier(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Iris Flower Species Prediction using SGD Classifier")
print("---------------------------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Get input from user
print("\nEnter Iris flower measurements:")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

# Prepare input
new_data = np.array([[sl, sw, pl, pw]])

new_data = scaler.transform(new_data)

# Predict species
prediction = model.predict(new_data)

print("\nPredicted Iris Species:",
      iris.target_names[prediction[0]])
```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="1511" height="370" alt="image" src="https://github.com/user-attachments/assets/c48bca28-5835-437a-b82f-d80307c7bbd7" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
