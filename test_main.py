from fastapi.testclient import TestClient
from main import app, Features

client = TestClient(app)


def test_api_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()['message'] == "Greetings! Welcome to this project!"


def test_target_negative_class():
    data = Features().dict(by_alias=True)
    response = client.post("/predictions", json=data)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Salary <= 50k'}


def test_target_positive_class():
    data = Features().dict(by_alias=True)
    data['age'] = 46
    data['workclass'] = ' Private'
    data['fnlgt'] = 117849
    data['education'] = ' Some-college'
    data['education_num'] = 10
    data['marital_status'] = " Married-civ-spouse"
    data['occupation'] = ' Sales'
    data['relationship'] = " Husband"
    data['race'] = ' White'
    data["sex"] = ' Male'
    data['capital_gain'] = 15024
    data['capital_loss'] = 0
    data['hours_per_week'] = 40
    data['native-country'] = ' United-States'

    response = client.post("/predictions", json=data)
    assert response.status_code == 200
    assert response.json() == {'predict': 'Salary > 50k'}
