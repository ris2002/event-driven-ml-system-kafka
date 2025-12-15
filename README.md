# API-Anomaly-Detection-MLOPS
This project is the the follow on (https://github.com/ris2002/Fraud-Credit-Card-detection-MLOPS.git) I had some errors in that project and my goal is to implemnnet the same architeccture from that project.
The data set hhas been obtained from Kaggle site (https://www.kaggle.com/datasets/josereimondez/fake-jobs-posting-detection)
## Challenges
One of the biggest challenge is this dataset contains a lot of null values. I had to come up with differrent startegies to work them out. 
* I have previously thought of placcig the na values with the mode value but sice it might give out some bias I have refreained from it
* I have columns which are mixed kike some col are numericcal, categorical ad some are sentences
* There is a challenge to vecctorize the sentences and one hot encode the categorical data
## OHE ONE HOT ENCODING
One-Hot Encoding is a method to convert categorical variables into numeric vectors so machine learning models can process them.
Each unique category in a column becomes a new binary column.
A value of 1 indicates that the row belongs to that category, and 0 means it does not.
Prevents ML models from assuming any order in categorical data.
Example
Original employment_type column:
employment_type
Full-time
Part-time
Contract

After One-Hot Encoding:

Full-time	Part-time	Contract
1	0	0
0	1	0
0	0	1

Works well with linear models, tree-based models, and other ML algorithms.

### What if the column already has 0/1?

If your column is already numeric (0/1), you usually do NOT need OHE.

The column is already binary, so the model can use it directly.

## Column Transformer
ColumnTransformer is a tool in scikit-learn that allows you to apply different preprocessing steps to different columns in a dataset simultaneously.

Useful when your dataset has mixed data types: text, categorical, numeric.

Automatically combines the outputs into a single feature matrix for ML models.

How it works

You define a list of transformers in the form:

('name', transformer_object, columns_to_apply)


Example:

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=5000), 'all_text'),  # apply TF-IDF to text column
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['employment_type','industry'])  # One-Hot for categorical
    ],
    remainder='drop'
)


fit on training data and transform test/deploy data to ensure consistent preprocessing.

Combines text vectorization and categorical encoding in a single step.

Benefits

Simplifies preprocessing for datasets with mixed features.

Ensures ML models receive a single numeric feature matrix.

Reduces manual concatenation or errors in feature preparation.