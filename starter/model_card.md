# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
### Overview
This is a scikit learn model which aims to classify whether or not an individual has an income of over $50,000 based on various demographic features such as work class, education, marital status, occupation, relationship, sex, race, and native country. The model is trained on UCI Census Income Dataset. 

### Version 
Version number: 1.0.0

## Intended Use
- Project is intended for research purposes of applying mlops concepts
- Model is intended to be used to determine what features impacts the income of a person.
- Model is intended to determine underprivileged employers.

## Training Data
- Census Income from UCI
- Encoded the categories using `OneHotEncoder` and target variable with `LabelBinarizer`
- Dropped the `education` column because it is already available encoded in the `education-num` column

## Evaluation Data
- Splitting the train data using sklearn `train_test_split` with `random_state=143` and  `test_size=0.2`.

## Metrics
- Precision: 85.71%
- Recall: 31.21%
- Fbeta: 45.76%

## Ethical Considerations
- Data is open sourced on UCI machine learning repository for educational purposes.

## Caveats and Recommendations
- Optimize hyperparameters and try other model architecture