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
# TODO: IGRAJ SE SA PARAMETRIMA MAJMUNE
# TODO: DODAJ MOZDA I NEKE DRUGE KLASIFIKATORE MOZDA JE TO PAMETNO
# TODO: DODAJ JOS NEKE FEATURE I MOZDA IZBACI NEKE PROBAJ PAR VARIJANTI, TREBAO BI MOZDA PROVERITI STA KOJI FEATURE PREDSTAVLJA
# TODO: STA RADI SERFEZE, TESTIRA KOD
param_grid = {
    'kernel': ['rbf'],
    'C': [1, 1, 10],
    'gamma': [0.00001, 0.4, 3]
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