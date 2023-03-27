# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml.data import process_data
import ml.model as mdl
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
logger.info("Extract data")
data = pd.read_csv('../data/census.csv')
data.columns = [col.strip() for col in data.columns]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Split data")
train, test = train_test_split(data, 
                               test_size=0.20, 
                               random_state=143)

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
logger.info("Process train data")
X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label='salary', 
        training=True
    )

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
logger.info("Train model")
model = mdl.train_model(X_train, y_train)

fnames = ['model', 'encoder', 'label_binarizer']
logger.info("Save models")
for fname, object in zip(fnames, [model, encoder, lb]):
    pickle.dump(object, open(f'../model/{fname}.pickle', 'wb'))

y_pred = mdl.inference(model, X_test)

logger.info("Evaluate model")
precision, recall, fbeta = mdl.compute_model_metrics(y_test, y_pred)

logger.info(f'''Precision:{precision}
            Recall:{recall}
            fbeta:{fbeta}''')

logger.info("Evaluate metrics for data slices")
metrics_category = mdl.evaluate_model_slice_data(
    model, data, cat_features, encoder, lb
)
