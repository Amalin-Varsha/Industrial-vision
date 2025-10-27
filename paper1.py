import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

# Load dataset
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.Series(cancer_data.target)

print("Breast Cancer Dataset Features:")
print(X.head())
print("\nTarget Names:")
print(cancer_data.target_names)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, X_test, y_test, model_name):
   
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_mat}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['malignant', 'benign']))


print("\nTraining Logistic Regression (Baseline)...")
lr_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression (Baseline)")


print("\nTraining Lasso (L1) Logistic Regression...")
lasso_lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso_lr_model.fit(X_train_scaled, y_train)
evaluate_model(lasso_lr_model, X_test_scaled, y_test, "Lasso (L1) Logistic Regression")


print("\nLasso Coefficients:")
for feature, coef in zip(cancer_data.feature_names, lasso_lr_model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


print("\nTraining Ridge (L2) Logistic Regression...")
ridge_lr_model = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=42)
ridge_lr_model.fit(X_train_scaled, y_train)
evaluate_model(ridge_lr_model, X_test_scaled, y_test, "Ridge (L2) Logistic Regression")


print("\nRidge Coefficients:")
for feature, coef in zip(cancer_data.feature_names, ridge_lr_model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


print("\nTraining Support Vector Machine (SVM) with RBF kernel...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  
svm_model.fit(X_train_scaled, y_train)
evaluate_model(svm_model, X_test_scaled, y_test, "Support Vector Machine (RBF Kernel)")


print("\nTraining K-Nearest Neighbors (KNN)...")
knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train_scaled, y_train)
evaluate_model(knn_model, X_test_scaled, y_test, "K-Nearest Neighbors (KNN)")


# --- Add the Decision Tree model here ---
print("\nTraining Decision Tree Classifier...")
dt_model = DecisionTreeClassifier('gini',random_state=42)
dt_model.fit(X_train_scaled, y_train)
evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree Classifier")

