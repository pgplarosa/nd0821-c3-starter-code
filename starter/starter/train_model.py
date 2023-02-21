# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
import pandas as pd
import pickle

# Add code to load in the data.
data = pd.read_csv('../data/census.csv')
data.columns = [col.strip() for col in data.columns]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label='salary', training=True
    )

# Train and save a model.
model = train_model(X_train, y_train)

fnames = ['model', 'encoder', 'label_binarizer']
for fname, object in zip(fnames, [model, encoder, lb]):
    pickle.dump(object, open(f'../model/{fname}.pickle', 'wb'))