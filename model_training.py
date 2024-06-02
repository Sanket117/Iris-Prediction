# model_training.py

import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the preprocessed data
X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(model, 'iris_model.pkl')
