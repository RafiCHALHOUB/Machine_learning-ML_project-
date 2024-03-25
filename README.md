# ProjectML Readme

## Overview

This repository contains code for a machine learning project focused on gene expression analysis using diverse techniques, including feature selection, dimensionality reduction, logistic regression, tree-based models (XGBoost), and neural networks implemented using the Keras library.

## Project Structure

- **functions.py**: Contains custom functions used in the project.
- **data.csv**: Gene expression data in CSV format.
- **labels.csv**: Labels associated with the gene expression data.

## Setup and Dependencies

Ensure you have the required dependencies installed:

```bash
pip install pandas scikit-learn seaborn matplotlib xgboost keras
```

## Usage

1. **Load Data and Labels:**

   - Import necessary libraries.
   - Load gene expression data from `data.csv`.
   - Load labels from `labels.csv`.

2. **Data Exploration and Cleaning:**

   - Remove columns with all zero values.
   - Apply variance threshold.

3. **Dimensionality Reduction (PCA):**

   - Standardize data.
   - Apply Principal Component Analysis (PCA).

4. **Visualize PCA Results:**

   - Create a scatter plot to visualize PCA results.

5. **Logistic Regression:**

   - Binarize labels.
   - Split data into training and testing sets.
   - Train a logistic regression model.
   - Evaluate the model and visualize the ROC curve.

6. **Learning Curves:**

   - Plot learning curves for logistic regression, XGBoost, and neural networks.

7. **XGBoost Model:**

   - Train a tree-based XGBoost model.
   - Evaluate the XGBoost model and perform cross-validation.

8. **Neural Network:**

   - Initialize and train a neural network.
   - Experiment with different configurations for neural networks.

## Acknowledgments

- The code in this repository utilizes various machine learning libraries and techniques. Special thanks to the developers and contributors of scikit-learn, Keras, XGBoost, and other open-source projects.

Feel free to experiment with different parameters, models, and techniques to enhance the performance of your gene expression analysis.