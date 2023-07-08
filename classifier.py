import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


csv_file = './features.csv'

df = pd.read_csv(csv_file)

X = df.drop(['genre'], axis=1)
y = df['genre']
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'kernel': ['rbf'],
    'C': [1, 1, 10],
}
svm = SVC()

print("Performing grid search...")
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
best_params = grid_search.best_params_

print("Training the SVM...")
svm_best = SVC(**best_params)
svm_best.fit(X_train, y_train)


decision_scores = svm_best.decision_function(X_test)


top_3_predictions = decision_scores.argsort(axis=1)[:, ::-1][:, :3]


correct_predictions = sum(true_label in top_3 for true_label, top_3 in zip(y_test, top_3_predictions))
top_3_accuracy = correct_predictions / len(y_test)
print("Top 3 Accuracy:", top_3_accuracy)


top_2_predictions = decision_scores.argsort(axis=1)[:, ::-1][:, :2]

correct_predictions = sum(true_label in top_2 for true_label, top_2 in zip(y_test, top_2_predictions))
top_2_accuracy = correct_predictions / len(y_test)
print("Top 2 Accuracy:", top_2_accuracy)

accuracy = svm_best.score(X_test, y_test)
print("Accuracy:", accuracy)