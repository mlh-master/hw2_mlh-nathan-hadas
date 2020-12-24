import pandas as pd
import numpy as np
import os
import pathlib as path
from sklearn.model_selection import train_test_split

directory = r'C:\Users\Nathan\PycharmProjects\ML_in_Healthcare_Winter2021\HW2'
# file_path = path.cwd().joinpath('HW2_data.csv')
file_path = os.path.join(directory, 'HW2_data.csv')
file = pd.read_csv(file_path)

missing_data = file[file.isnull().any(axis=1)]
pos_nan = missing_data[missing_data['Diagnosis'] == 'Positive']
positive_val = file[file['Diagnosis'] == 'Positive']

# Our data consists mainly of 'Positive' samples, thus in order to reduce unnecessary/
# data distortion, we decided to remove missing data labeled 'Positive'

clean_data = file.drop(pos_nan.index)

# All nan samples are now necessarily labeled as 'Negative'

# We decided to complete the missing samples by the distribution of each/
# feature for samples tagged as a 'Negative'

neg_data = clean_data[clean_data['Diagnosis'] == 'Negative']
for key in neg_data.keys():
    temp_prob = neg_data[key].value_counts()
    dominant_val = temp_prob[temp_prob == max(temp_prob)].index[0]

    clean_data[key] = [val if not pd.isna(val) else dominant_val for val in clean_data[key]]


# split data function: split the data randomly to train-test with test fraction
y = clean_data['Diagnosis'].values
X = clean_data.drop(['Diagnosis'], axis=1)

X_train, x_test, Y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10, stratify=y)
