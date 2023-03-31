import requests
import json

data = dict()
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

print('input_data: ', data)

resp1 = requests.post('https://income-classifier.onrender.com/predictions', 
              data=json.dumps(data))
print('status code: ', resp1.status_code)
print('predictions: ', resp1.json())

data = dict()
data['age'] = 46
data['workclass'] = 'State-gov'
data['fnlgt'] = 77516
data['education'] = 'Bachelors'
data['education_num'] = 13
data['marital_status'] = 'Never-married'
data['occupation'] = 'Adm-clerical'
data['relationship'] = 'Not-in-family'
data['race'] = 'White'
data["sex"] = 'Male'
data['capital_gain'] = 2174
data['capital_loss'] = 0
data['hours_per_week'] = 40
data['native-country'] = ' United-States'
print('input_data: ', data)
resp2 = requests.post('https://income-classifier.onrender.com/predictions', 
              data=json.dumps(data))


print('status code: ', resp2.status_code)
print('predictions: ', resp2.json())


