import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Contoh hasil prediksi (simulasi)
y_test = np.random.randint(0, 10, size=1000)
y_pred = np.random.randint(0, 10, size=1000)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - MNIST CNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
report = classification_report(y_test, y_pred, target_names=[f'Digit {i}' for i in range(10)])
print(report)

# Simpan visualisasi
plt.savefig("mnist_cnn_evaluation.png")

# Simpan laporan klasifikasi ke file
with open("mnist_cnn_report.txt", "w") as f:
    f.write(report)

print("Visualisasi evaluasi disimpan sebagai 'mnist_cnn_evaluation.png'")
print("Laporan klasifikasi disimpan sebagai 'mnist_cnn_report.txt'")

