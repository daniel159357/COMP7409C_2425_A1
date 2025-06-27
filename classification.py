import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Load and preprocess the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, header=None, names=column_names)

# Filter for Setosa and Versicolor
df = df[df['class'].isin(['Iris-setosa', 'Iris-versicolor'])]

# Select only sepal_length and petal_length
X = df[['sepal_length', 'petal_length']].values

# Convert labels to -1 (Setosa) and 1 (Versicolor)
y = df['class'].map({'Iris-setosa': -1, 'Iris-versicolor': 1}).values

# Display modified data
print("\nModified Data (First 5 samples):")
print(df.head())

# Train the Perceptron
ppn = Perceptron(eta=0.1, nepoch=10)
ppn.fit(X, y)

# Plot error progression
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.title('Perceptron Training: Errors vs. Epochs')
plt.xticks(range(1, 11))
plt.yticks(np.arange(0, 3.5, 0.5))
plt.grid(True)
plt.show()

# If plt.show() fails, save the image:
plt.savefig('perceptron_errors.png')
print("Plot saved as 'perceptron_errors.png'")