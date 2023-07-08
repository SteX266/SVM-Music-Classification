import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the 'genre' column

# Assuming you have a DataFrame called 'df' with the features and target variable
# features = {"genre": [], "zero_crossing_rate": [], "spectral_centroid": []}

csv_file = './features.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)
# Convert the features dictionary to a DataFrame

# Split the dataset into training and testing sets
X = df.drop(['genre'], axis=1)
y = df['genre']
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'kernel': ['rbf'],
    'C': [1, 1, 10],
}
# Create the SVM classifier
svm = SVC()

# Perform grid search to determine the best hyperparameters
print("Performing grid search...")
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the SVM with the best hyperparameters
print("Training the SVM...")
svm_best = SVC(**best_params)
svm_best.fit(X_train, y_train)


decision_scores = svm_best.decision_function(X_test)

print("Decision scores:")
print(decision_scores)

# Sort the decision scores in descending order for each instance
top_3_predictions = decision_scores.argsort(axis=1)[:, ::-1][:, :3]
print("top 3 predictions")
print(top_3_predictions)


# Calculate the top 3 accuracy
correct_predictions = sum(any(true_label in top_3 for top_3 in top_3_predictions) for true_label in y_test)
top_3_accuracy = correct_predictions / len(y_test)
print("Top 3 Accuracy:", top_3_accuracy)


top_2_predictions = decision_scores.argsort(axis=1)[:, ::-1][:, :2]
print("top 2 predictions")

print(top_2_predictions)

# Calculate the top 2 accuracy
correct_predictions = sum(any(true_label in top_2 for top_2 in top_2_predictions) for true_label in y_test)
top_2_accuracy = correct_predictions / len(y_test)
print("Top 2 Accuracy:", top_2_accuracy)

# Evaluate the SVM on the testing set
accuracy = svm_best.score(X_test, y_test)
print("Accuracy:", accuracy)