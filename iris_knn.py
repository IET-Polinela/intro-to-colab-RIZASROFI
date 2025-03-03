from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
iris = load_iris()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualisasi hasil evaluasi
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Simpan model
joblib.dump(knn, "iris_knn.pkl")
print("Model disimpan sebagai 'iris_knn.pkl'")

# Simpan visualisasi
plt.savefig("iris_knn_evaluation.png")
print("Visualisasi evaluasi disimpan sebagai 'iris_knn_evaluation.png'")

# Simpan laporan klasifikasi ke file
with open("iris_knn_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Laporan klasifikasi disimpan sebagai 'iris_knn_report.txt'")
