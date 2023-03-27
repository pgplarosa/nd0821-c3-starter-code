from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(
        random_state=143, max_depth=4)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
      using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_model_slice_data(
        model, data, categorical_features_list,
        encoder, lb):

    preds = {}
    for column in categorical_features_list:
        for feature in data[column].unique():
            data_temp = data[data[column] == feature]
            X_feature, y_feature, encoder, lb = process_data(
                data_temp,
                categorical_features=categorical_features_list,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            predict = inference(model, X_feature)
            precision, recall, fbeta = compute_model_metrics(
                y_feature, predict)

            preds[feature] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta}

    return preds
