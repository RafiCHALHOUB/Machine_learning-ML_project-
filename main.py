from keras.utils import plot_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score
from functions import *
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout
# Load data
data = pd.read_csv('/home/rafi/PycharmProjects/ProjectML/gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531/data.csv')

# Load labels
labels = pd.read_csv('/home/rafi/PycharmProjects/ProjectML/gene+expression+cancer+rna+seq/TCGA-PANCAN-HiSeq-801x20531/labels.csv')

# Display information about the data DataFrame
print(data.info())
# Display the first few rows of the data DataFrame
print(data.head())
# Display information about the labels DataFrame
print(labels.info())

# Display the first few rows of the labels DataFrame
print(labels.head())

# Remove columns where all values are zero (including floating-point zeros)
data_cleaned = data.loc[:, (data != 0).any(axis=0)]

# Check the new shape of the DataFrame
print(f"Original shape: {data.shape}, Shape after removing zero columns: {data_cleaned.shape}")

# Identifier les colonnes avec des zéros
zero_columns = data.columns[(data == 0).all()]

# Afficher le nombre de colonnes avec des zéros
print(f"Nombre de colonnes avec des zéros : {len(zero_columns)}")
print("Colonnes avec des zéros :", zero_columns)
print(data_cleaned.info)


# Calculate summary statistics for each feature
summary_statistics = data_cleaned.describe()

# Display the summary statistics
print(summary_statistics)

# Set the 'Unnamed: 0' column (sample names) as the index
data_cleaned.set_index('Unnamed: 0', inplace=True)

# Calculate standard deviation before variance threshold
std_before = data_cleaned.std()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bar chart for comparison
std_before.plot(kind='bar', color='blue', alpha=0.7, label='Before Variance Threshold')

# Set labels and title
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation Before Variance Threshold')

# Add legend
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()

# Create a histogram for the distribution of standard deviations
plt.figure(figsize=(12, 6))
plt.hist(std_before, bins=50, color='red', edgecolor='black', alpha=0.7)
plt.title('Distribution of Standard Deviations Before Variance Threshold')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Créer une instance de VarianceThreshold avec le seuil choisi
selector = VarianceThreshold(threshold=0.3)

# Appliquer la sélection des caractéristiques sur les données
data_selected = selector.fit_transform(data_cleaned)

# Obtenir les indices des caractéristiques sélectionnées
selected_indices = selector.get_support(indices=True)

# Afficher le nombre de caractéristiques sélectionnées
print(f"Nombre de caractéristiques sélectionnées : {data_selected.shape[1]}")

# Afficher les indices des caractéristiques sélectionnées
print("Indices des caractéristiques sélectionnées :", selected_indices)

# Standardize the data before applying PCA
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_selected)

# Specify the number of components you want to keep (adjust as needed)
n_components = 2

# Create PCA instance
pca = PCA(n_components=n_components)

# Apply PCA to the standardized data
data_pca = pca.fit_transform(data_standardized)

# Display the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Assuming data_pca is the result of PCA transformation and labels is a list/array containing class labels
plt.figure(figsize=(10, 8))

# Use Seaborn for scatter plot with different colors for each category
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels['Class'], palette='viridis', edgecolor='w', alpha=0.7)

# Customize the plot
plt.title('Scatter Plot of First Two Principal Components with Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Class Label')

plt.show()

'''Logistic Regression'''
# Assume labels['Class'] contains the original multiclass labels

# Binarize the labels
binarized_labels = label_binarize(labels['Class'], classes=['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD'])

# If a sample belongs to 'BRCA', set the label to 1, otherwise, set it to 0
binary_labels = (binarized_labels[:, 0] == 1).astype(int)

# Split the data and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_selected, binary_labels, test_size=0.2, random_state=10000)

# Create a logistic regression model
logreg_model = LogisticRegression(random_state=42)

# Train the model on the training data
logreg_model.fit(X_train, y_train)

# Predict the labels on the test data
y_pred = logreg_model.predict(X_test)

# Create a crosstab for better visualization
crosstab = pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted'), margins=True)

# Display the crosstab
print(crosstab)


# Assuming y_test and y_pred are your actual and predicted labels, respectively
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Accuracy on the test set: {accuracy:.2%}")


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, logreg_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



'''Learning curve on logistic regression model'''
MyFunctions.logreg_learning_curve(logreg_model, X_train, y_train, X_test, y_test)


'''Tree-based model'''
# Convert data to DMatrix format
dtrain = xgboost.DMatrix(data=X_train, label=y_train)
dtest = xgboost.DMatrix(data=X_test)

# Define XGBoost parameters
params = {'objective': 'binary:logistic', 'verbosity': 0, 'alpha': 0.1,'lambda': 1.0  }

# Train the XGBoost model
bst = xgboost.train(params=params, dtrain=dtrain)

# Make predictions on the test set
preds = bst.predict(data=dtest)

# Convert probabilities to binary predictions (assuming binary classification)
binary_preds = (preds > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, binary_preds)
precision = precision_score(y_test, binary_preds)
recall = recall_score(y_test, binary_preds)
f1 = f1_score(y_test, binary_preds)

print(f"Accuracy on the test set: {accuracy:.2%}")
print(f"Precision on the test set: {precision:.2%}")
print(f"Recall on the test set: {recall:.2%}")
print(f"F1-score on the test set: {f1:.2%}")


# Usage example:
MyFunctions.xgboost_learning_curve(bst, X_train, y_train, X_test, y_test, step_size=100)



# Perform cross-validation
cv_results = xgboost.cv(params=params, dtrain=dtrain, num_boost_round=10, nfold=5, metrics=['error'])

# Display the cross-validation results
print("Results with cross-validation:")
print(cv_results)

# Identify the index of the row with the lowest test error
optimal_round_index = cv_results['test-error-mean'].idxmin()

# Extract the optimal number of boosting rounds
optimal_num_boost_round = optimal_round_index + 1

print(f"Optimal Number of Boosting Rounds: {optimal_num_boost_round}")

# Retrain the XGBoost model with the optimal number of boosting rounds
bst_with_cv = xgboost.train(params=params, dtrain=dtrain, num_boost_round=optimal_num_boost_round)

# Make predictions on the test set with the retrained model
preds_with_cv = bst_with_cv.predict(data=dtest)

# Convert probabilities to binary predictions
binary_preds_with_cv = (preds_with_cv > 0.5).astype(int)

# Evaluate the model with cross-validation
accuracy_with_cv = accuracy_score(y_test, binary_preds_with_cv)
precision_with_cv = precision_score(y_test, binary_preds_with_cv)
recall_with_cv = recall_score(y_test, binary_preds_with_cv)
f1_with_cv = f1_score(y_test, binary_preds_with_cv)

print("Results with cross-validation (retrained model):")
print(f"Accuracy: {accuracy_with_cv:.2%}")
print(f"Precision: {precision_with_cv:.2%}")
print(f"Recall: {recall_with_cv:.2%}")
print(f"F1-score: {f1_with_cv:.2%}")

MyFunctions.xgboost_learning_curve(bst_with_cv, X_train, y_train, X_test, y_test, step_size=100)

'''Neural Network'''
def init_model():
    # Build the neural network
    init = 'random_uniform'
    input_layer = Input(shape=(16298,))
    mid_layer = Dense(550, activation='relu', kernel_initializer=init)(input_layer)
    mid_layer_2 = Dense(325, activation='relu', kernel_initializer=init)(mid_layer)
    mid_layer_3 = Dense(256, activation='relu', kernel_initializer=init)(mid_layer_2)
    mid_layer_4 = Dense(128, activation='relu', kernel_initializer=init)(mid_layer_3)
    mid_layer_5 = Dense(64, activation='relu', kernel_initializer=init)(mid_layer_4)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=init)(mid_layer_5)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Initialize the model
sgd_model = init_model()
# Call the learning curve function
MyFunctions.neuro_learning_curve(sgd_model, X_train, y_train, X_test, y_test)
plot_model(sgd_model, to_file='neural_network.png', show_shapes=True, show_layer_names=True)


# Evolving epochs number
MyFunctions.neuro_learning_curve(sgd_model, X_train, y_train, X_test, y_test,epochs=180)

#Using Adam optimizer
def adam_model():
    init = 'random_uniform'
    input_layer = Input(shape=(16298,))
    mid_layer = Dense(550, activation='relu', kernel_initializer=init)(input_layer)
    mid_layer_2 = Dense(325, activation='relu', kernel_initializer=init)(mid_layer)
    mid_layer_3 = Dense(256, activation='relu', kernel_initializer=init)(mid_layer_2)
    mid_layer_4 = Dense(128, activation='relu', kernel_initializer=init)(mid_layer_3)
    mid_layer_5 = Dense(64, activation='relu', kernel_initializer=init)(mid_layer_4)
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=init)(mid_layer_5)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# adam : doesn't reduce alpha each time
adam_model = adam_model()
MyFunctions.neuro_learning_curve(adam_model,X_train, y_train, X_test, y_test)
plot_model(adam_model, to_file='neural_network_adam.png', show_shapes=True, show_layer_names=True)

def adam_model_with_dropout():
    init = 'random_uniform'
    input_layer = Input(shape=(16298,))
    mid_layer = Dense(550, activation='relu', kernel_initializer=init)(input_layer)
    mid_layer_2 = Dense(325, activation='relu', kernel_initializer=init)(mid_layer)
    mid_layer_2 = Dropout(0.5)(mid_layer_2)  # Add a dropout layer with a dropout rate (e.g., 0.5)
    mid_layer_3 = Dense(256, activation='relu', kernel_initializer=init)(mid_layer_2)
    mid_layer_3 = Dropout(0.5)(mid_layer_3)  # Add a dropout layer with a dropout rate
    mid_layer_4 = Dense(128, activation='relu', kernel_initializer=init)(mid_layer_3)
    mid_layer_4 = Dropout(0.5)(mid_layer_4)  # Add a dropout layer with a dropout rate
    mid_layer_5 = Dense(64, activation='relu', kernel_initializer=init)(mid_layer_4)
    mid_layer_5 = Dropout(0.5)(mid_layer_5)  # Add a dropout layer with a dropout rate
    output_layer = Dense(1, activation='sigmoid', kernel_initializer=init)(mid_layer_5)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

adam_model_with_dropout_fn = adam_model_with_dropout()
MyFunctions.neuro_learning_curve(adam_model_with_dropout_fn,X_train, y_train, X_test, y_test)





