import pytest
import pandas as pd


@pytest.fixture(scope="module")
def dataframe():
    path = 'starter/data/census.csv'
    return pd.read_csv(path)


def test_extract_data():
    path = 'starter/data/census.csv'

    try:
        dataframe = pd.read_csv(path)
    except FileNotFoundError as err:
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] == 15
    except AssertionError as err:
        raise err


def test_features(dataframe):
    data = dataframe.copy()
    data.columns = [col.strip() for col in data.columns]
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
    assert len(set(cat_features)
               - set(data.columns.tolist())) == 0


def test_salary_cat(dataframe):
    data = dataframe.copy()
    salary_cat = data[' salary'].value_counts().index.tolist()
    assert len(salary_cat) == 2
