# Iris Classification with Perceptron

This project implements a Perceptron classifier to distinguish between two classes of Iris flowers (Setosa and Versicolor) using their sepal length and petal length measurements.

## Project Overview

The project demonstrates the implementation and application of the Perceptron learning algorithm, one of the fundamental building blocks of neural networks, on the classic Iris dataset.

### Features

- Implementation of a custom Perceptron classifier
- Training on the Iris dataset (binary classification)
- Visualization of the learning process through error plots
- Data preprocessing and feature selection

## Project Structure

- `perceptron.py` - Contains the Perceptron class implementation
- `classification.py` - Main script for data loading, preprocessing, and model training
- `perceptron_errors.png` - Generated plot showing the training errors across epochs

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Dataset

The project uses the famous Iris dataset from the UCI Machine Learning Repository. The dataset is automatically downloaded from:
```
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

Only two classes (Setosa and Versicolor) are used for binary classification, and two features (sepal length and petal length) are selected for training.

## Usage

To run the classification:

```bash
python classification.py
```

This will:
1. Load and preprocess the Iris dataset
2. Train the Perceptron classifier
3. Generate and save a plot showing the number of errors per epoch during training

## Output

The program outputs:
- The first 5 samples of the preprocessed dataset
- A plot showing the training errors across epochs (saved as 'perceptron_errors.png')

## Implementation Details

- Learning rate (η): 0.1
- Number of epochs: 10
- Features used: sepal length and petal length
- Classes: Iris-setosa (-1) and Iris-versicolor (1)
