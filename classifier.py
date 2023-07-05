import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Assuming you have a DataFrame called 'df' with the features and target variable
# features = {"genre": [], "zero_crossing_rate": [], "spectral_centroid": []}

csv_file = './features.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)
# Convert the features dictionary to a DataFrame

# Split the dataset into training and testing sets
X = df[['zero_crossing_rate', 'spectral_centroid']]
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'C': [1, 10, 12],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 1.2]
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

# Evaluate the SVM on the testing set
accuracy = svm_best.score(X_test, y_test)
print("Accuracy:", accuracy)