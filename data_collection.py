# data_collection.py

import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Create a DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
data['target_names'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Save to CSV
data.to_csv('iris.csv', index=False)
