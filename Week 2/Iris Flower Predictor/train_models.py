from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import os

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
pickle.dump(log_model, open("models/logistic.pkl", "wb"))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
pickle.dump(knn_model, open("models/knn.pkl", "wb"))

# SVM
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)
pickle.dump(svm_model, open("models/svm.pkl", "wb"))

print("All models trained and saved successfully!")
