import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('iris.csv', header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Encode the target labels
data['species'] = data['species'].astype('category').cat.codes

# Split the data into features and target
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the preprocessed data and scaler
joblib.dump((X_train, X_test, y_train, y_test), 'preprocessed_data.pkl')
joblib.dump(scaler, 'scaler.pkl')
