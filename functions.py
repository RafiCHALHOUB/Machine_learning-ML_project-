import xgboost
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class MyFunctions:
    def xgboost_learning_curve(self, X_train, y_train, X_test, y_test, step_size=100):
        examples = []
        training_accuracies = []
        testing_accuracies = []

        for i in range(10, len(X_train) + 1, step_size):
            # Train the model using XGBoost's train method
            dtrain = xgboost.DMatrix(data=X_train[:i], label=y_train[:i])
            params = {'objective': 'binary:logistic', 'verbosity': 0}
            self = xgboost.train(params=params, dtrain=dtrain)

            # Calculate the training accuracy by predicting on the entire training set
            training_preds = self.predict(xgboost.DMatrix(data=X_train))
            training_accuracy = accuracy_score(
                (training_preds > 0.5).astype(int), y_train)

            # Calculate the testing accuracy by predicting on the entire testing set
            testing_preds = self.predict(xgboost.DMatrix(data=X_test))
            testing_accuracy = accuracy_score(
                (testing_preds > 0.5).astype(int), y_test)

            # Append the training and testing accuracies to their respective lists
            training_accuracies.append(training_accuracy)
            testing_accuracies.append(testing_accuracy)
            examples.append(i)

        learning_curve_df = pd.DataFrame(
            {'examples': examples, 'training_accuracy': training_accuracies, 'testing_accuracy': testing_accuracies})

        plt.figure(figsize=(10, 6))
        plt.plot(learning_curve_df['examples'], learning_curve_df['training_accuracy'], label='Training Accuracy',
                 marker='o', markersize=0.1)
        plt.plot(learning_curve_df['examples'], learning_curve_df['testing_accuracy'], label='Testing Accuracy',
                 marker='o', markersize=0.1)
        plt.title('Learning Curve')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        return learning_curve_df

    @staticmethod
    def logreg_learning_curve(model, X_train, y_train, X_test, y_test, step_size=100):
        examples = []
        training_accuracies = []  # List to store training accuracies at each step
        testing_accuracies = []  # List to store testing accuracies at each step

        # Iterate over a cumulative range from step_size to the length of the training set
        for i in range(10, len(X_train) + 1, step_size):
            # Train the model on the first i samples of the training data
            model.fit(X_train[:i], y_train[:i])

            # Calculate the training accuracy by predicting on the entire training set
            training_accuracy = accuracy_score(model.predict(X_train), y_train)

            # Calculate the testing accuracy by predicting on the entire testing set
            testing_accuracy = accuracy_score(model.predict(X_test), y_test)

            # Append the training and testing accuracies to their respective lists
            training_accuracies.append(training_accuracy)
            testing_accuracies.append(testing_accuracy)
            examples.append(i)

        # Create a dataframe
        learning_curve_df = pd.DataFrame(
            {'examples': examples, 'training_accuracy': training_accuracies, 'testing_accuracy': testing_accuracies})

        # Plot the learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(learning_curve_df['examples'], learning_curve_df['training_accuracy'], label='Training Accuracy',
                 marker='o', markersize=0.1)
        plt.plot(learning_curve_df['examples'], learning_curve_df['testing_accuracy'], label='Testing Accuracy',
                 marker='o', markersize=0.1)
        plt.title('Learning Curve')
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        return learning_curve_df

    @staticmethod
    def neuro_learning_curve(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        training_accuracy = []
        testing_accuracy = []
        index = []

        for i in range(30, len(X_train), 20):
            # Model fit
            model.fit(X_train[:i], y_train[:i], batch_size=batch_size, epochs=epochs, verbose=0)
            # Prediction with training dataset
            Z_train = model.predict(X_train[:i])
            # Prediction with testing dataset
            Z_test = model.predict(X_test)
            prediction_train = np.round(Z_train)
            prediction_test = np.round(Z_test)
            # Accuracy
            training_accuracy.append(accuracy_score(np.round(y_train[:i]), prediction_train))
            testing_accuracy.append(accuracy_score(np.round(y_test), prediction_test))
            index.append(i)

        plt.plot(index, training_accuracy, label='Training')
        plt.plot(index, testing_accuracy, label='Testing')
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
