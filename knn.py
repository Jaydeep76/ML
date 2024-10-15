import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create the dataset (features and labels)
# Features: (x1, x2)
X = np.array([[2, 3], [1, 1], [5, 4], [3, 5], [6, 7], [7, 8], [1, 0], [4, 2]])
# Labels (target variable, e.g., class 0 or 1)
y = np.array([0, 0, 1, 0, 1, 1, 0, 1])

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize the K-NN classifier (using k=3 neighbors)
knn = KNeighborsClassifier(n_neighbors=1)

# Step 4: Train the model
knn.fit(X_train, y_train)

# Step 5: Predict the value of an unknown point
# Unknown point we want to predict (let's say the point is [4, 4])
unknown_point = np.array([[4, 4]])

# Step 6: Use the trained model to predict the class of the unknown point
predicted_class = knn.predict(unknown_point)
print(f"Predicted class for the point {unknown_point} is: {predicted_class[0]}")

# Step 7: Evaluate the accuracy of the model on the test data (optional)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the K-NN model: {accuracy * 100:.2f}%")
