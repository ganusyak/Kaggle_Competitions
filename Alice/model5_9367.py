from matplotlib import pyplot as plt
import seaborn as sns

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

###########
# FUNCTIONS
###########

def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver='lbfgs', max_iter=500).fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)

    return score



times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]

train_df = pd.read_csv('train_sessions.csv', index_col='session_id', parse_dates=times)
#test_df = pd.read_csv('test_sessions.csv', index_col='session_id', parse_dates=times)

# Sort the data by time
train_df = train_df.sort_values(by='time1')


train_df[sites] = train_df[sites].fillna(0).astype('int')
#test_df[sites] = test_df[sites].fillna(0).astype('int')

# Load websites dictionary
with open(r"site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
#print(u'Websites total:', sites_dict.shape[0])
#sites_dict.head()

# Our target variable
y_train = train_df['target'].values

# United dataframe of the initial data
#full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]

# small
train_df[sites].fillna(0).to_csv('train_sessions_text.txt', sep=' ', index=None, header=None)
#test_df[sites].fillna(0).to_csv('test_sessions_text.txt', sep=' ', index=None, header=None)


cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
with open('train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
#with open('test_sessions_text.txt') as inp_test_file: X_test = cv.transform(inp_test_file)
print(X_train.shape)

####################################
#   CROSSVALIDATION
####################################

time_split = TimeSeriesSplit(n_splits=10)
logit = LogisticRegression(C=1, random_state=17, solver='liblinear')

cv_scores = cross_val_score(logit, X_train, y_train, cv=time_split, scoring='roc_auc', n_jobs=-1)
print('Initial data')
print(cv_scores)
print(cv_scores.mean(), cv_scores.std())

# Placeholder for some new features
train_df_newfeatures = pd.DataFrame(index=train_df.index)

#   Feature number of urls in session < 10
#   EDA shows, that Alice almost always has 10 urls in session

train_df_newfeatures['sub10'] = (train_df[times].count(axis = 1) < 10) * 1 - 0.5

#    Day of week

train_df_newfeatures['dow'] = train_df['time1'].apply(lambda ts : ts.date().weekday())

#    Active days. Alice seems to be active at 0, 1, 3 and 4 days if week

train_df_newfeatures['dow'] = train_df['time1'].apply(lambda ts : ts.date().weekday())
train_df_newfeatures['active_days'] = (train_df_newfeatures['dow'].apply(lambda x : x in [0, 1, 3, 4]) ) * 1 - 0.5

#    Active hours. Alice active only at certain hours

train_df_newfeatures['hour'] = train_df['time1'].apply(lambda ts : ts.hour)
train_df_newfeatures['active_hours'] = (train_df_newfeatures['hour'].apply(lambda x : x in [12, 13, 16, 17, 18]) ) * 1 - 0.5

#    Session lenght in seconds

train_df_newfeatures['sesslen'] = (train_df[times].max(axis = 1) - train_df[times].min(axis = 1)).apply(lambda ts: round(ts.seconds))


####################################
#    Feature scaling
####################################

scaler = StandardScaler()
scaler.fit(train_df_newfeatures['dow'].values.reshape(-1, 1))
train_df_newfeatures['dow_scaled'] = scaler.fit_transform(train_df_newfeatures['dow'].values.reshape(-1,1))

scaler.fit(train_df_newfeatures['sesslen'].values.reshape(-1, 1))
train_df_newfeatures['sesslen_scaled'] = scaler.fit_transform(train_df_newfeatures['sesslen'].values.reshape(-1,1))

####################################
#    Adding new features to train dataset
####################################

X_train_new = csr_matrix(hstack([X_train, train_df_newfeatures[['dow', 'active_days', 'active_hours', 'sesslen']]]))

cv_scores = cross_val_score(logit, X_train_new, y_train, cv=time_split, scoring='roc_auc', n_jobs=-1)
print('num_urls + day of week')
print(cv_scores)
print(cv_scores.mean(), cv_scores.std())
