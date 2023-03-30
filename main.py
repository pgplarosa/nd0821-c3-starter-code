# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

from starter.starter.ml.data import process_data

app = FastAPI()

# Import model, encoder and lb
with open('starter/model/model.pickle', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

with open('starter/model/encoder.pickle', 'rb') as pickle_file:
    encoder = pickle.load(pickle_file)

with open('starter/model/label_binarizer.pickle', 'rb') as pickle_file:
    lb = pickle.load(pickle_file)

# categorical features
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Features to ingest the body from POST


class Features(BaseModel):
    age: int = 39
    workclass: str = 'State-gov'
    fnlgt: int = 77516
    education: str = 'Bachelors'
    education_num: int = 13
    marital_status: str = 'Never-married'
    occupation: str = 'Adm-clerical'
    relationship: str = 'Not-in-family'
    race: str = 'White'
    sex: str = 'Male'
    capital_gain: int = 2174
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = 'United-States'


@app.get("/")
async def welcome_user():
    return {"message":
            "Greetings! Welcome to this project!"}


@app.post("/predictions")
def predict_model(features: Features):
    data = pd.DataFrame(data=features.dict(by_alias=True), index=[0])
    print(data)
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb)

    predict = model.predict(X)

    if predict[0] == 1:
        predict = "Salary > 50k"
    else:
        predict = "Salary <= 50k"

    return {'predict': predict}
