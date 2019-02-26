# pylint: disable=unused-import
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from titanic_estimator import *

# read data from csv file
data = pd.read_csv('train.csv')

# add family column
for i in range(data.shape[0]):
    data.at[i, 'Family'] = data.at[i, 'Name'].split(', ')[0]

# add deck feature
for i in range(data.shape[0]):
    if pd.isnull(data.at[i, 'Cabin']):
        data.at[i, 'Deck'] = 'U'
    else:
        data.at[i, 'Deck'] = list(data.at[i, 'Cabin'])[0]


# add family members feature
data['FamilyMembers'] = data.groupby('Family')['Family'].transform('count')
data = data[['PassengerId', 'Family', 'Pclass',
             'Sex', 'Age', 'FamilyMembers',
             'Deck', 'Survived']]


# split data features from labels
train_df = data.sample(frac=0.9, random_state=0)
test_df = data.drop(train_df.index)

train_y = train_df.pop('Survived')
test_y = test_df.pop('Survived')
train_x = train_df
test_x = test_df

# train classifier
classifier.train(
    input_fn=lambda: input_fn(train_x, train_y, batch_size=100),
    steps=4501
)

# evaluate on test data
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, batch_size=100))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
