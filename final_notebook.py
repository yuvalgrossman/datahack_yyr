
# coding: utf-8

# <h1>imports</h1>

# In[219]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
from urllib import request
import json
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, Normalizer, Binarizer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from collections import Counter


# <h1>load data</h1>

# In[220]:


data_path = './data/'


# In[221]:


train_accounts = pd.read_csv(data_path + 'train_accounts.csv')#.sample(frac=0.2)
# train_users = pd.read_csv(data_path + 'train_users.csv')
# train_events = pd.read_csv(data_path + 'train_events.csv')
# train_subscriptions = pd.read_csv(data_path + 'train_subscriptions.csv')
test_accounts = pd.read_csv(data_path + 'test_accounts.csv')
# test_users = pd.read_csv(data_path + 'test_users.csv')
# test_events = pd.read_csv(data_path + 'test_events.csv')
# test_subscriptions = pd.read_csv(data_path + 'test_subscriptions.csv')


# In[222]:


accounts = pd.concat([train_accounts, test_accounts],sort=False)
print(f'accounts: {len(accounts)}')
#users = pd.concat([train_users, test_users],sort=False)
#events = pd.concat([train_events, test_events],sort=False)
#subscriptions = pd.concat([train_subscriptions, test_subscriptions],sort=False)


# <h1>feature engineering</h1>

# look at
# churn date
# user role
# domain

# In[223]:


# transform plan_id & utm_cluster_id to str since its categorical
accounts['plan_id'] = accounts['plan_id'].astype(str)
accounts['utm_cluster_id'] = accounts['utm_cluster_id'].astype(str)


# In[224]:


def clip_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_in.loc[df_in[col_name] > fence_high, col_name] = fence_high
    df_in.loc[df_in[col_name] < fence_low, col_name] = fence_low
    
def creating_time_features(data):
    time_between_created_trial = pd.to_datetime(data['trial_start']) - pd.to_datetime(data['created_at'])
    time_between_created_subscription = pd.to_datetime(data['subscription_started_at']) - pd.to_datetime(data['created_at'])
    time_between_trial_subscription = pd.to_datetime(data['subscription_started_at']) - pd.to_datetime(data['trial_start'])
    time_between_now_trial = datetime.now() - pd.to_datetime(data['trial_start'])
    time_between_now_subscription = datetime.now() - pd.to_datetime(data['subscription_started_at'])
    time_between_now_created = datetime.now() - pd.to_datetime(data['created_at'])
    time_between_now_churn = datetime.now() - pd.to_datetime(data['churn_date'])
    time_between_churn_subscription = pd.to_datetime(data['churn_date']) - pd.to_datetime(data['subscription_started_at'])

    data = data.assign(created_trial_delta=time_between_created_trial.apply(lambda x: (x.seconds//3600)))
    data = data.assign(created_subscription_delta=time_between_created_subscription.apply(lambda x: (x.seconds//3600)))
    data = data.assign(trial_subscription_delta=time_between_trial_subscription.apply(lambda x: (x.seconds//3600)))
    data = data.assign(now_trial_delta=time_between_now_trial.apply(lambda x: (x.seconds//3600)))
    data = data.assign(now_subscription_delta=time_between_now_subscription.apply(lambda x: (x.seconds//3600)))
    data = data.assign(now_created_delta=time_between_now_created.apply(lambda x: (x.seconds//3600)))
    data = data.assign(now_churn_delta=time_between_now_churn.apply(lambda x: (x.seconds//3600)))
    data = data.assign(churn_subscription_delta=time_between_churn_subscription.apply(lambda x: (x.seconds//3600)))
    data['is_subscription'] = (data.subscription_started_at.isna()).astype(int)
    data['is_churn'] = (data.churn_date.isna()).astype(int)
    return data

def creating_size_n_survey_features(data, bins, bins_labels):
    #clip_outlier(data,'company_size')
    data.loc[:,'avg_team_size'] = data[["min_team_size", "max_team_size"]].mean(axis=1)
    data['avg_team_size'].fillna(-1, inplace=True)
    data['avg_team_cat'] = pd.cut(data['avg_team_size'], bins=bins, labels=bins_labels)
    data['avg_team_cat'] = data['avg_team_cat'].astype(str)
    data['survey_answers'] = data[['company_size','max_team_size','min_team_size','user_goal','user_description','team_size']].isna().sum(axis=1)
    data['survey_did_answer'] = data['survey_answers']
    return data


# In[225]:


# creating size & survey features
bins = sorted((list(train_accounts["max_team_size"].value_counts().index) + [-1.1, -1, ]))
bins_labels = [str(b) for b in bins[1:]]

accounts = creating_time_features(accounts)
accounts = creating_size_n_survey_features(accounts, bins, bins_labels)
accounts['country_counts'] = accounts.groupby('country')['country'].transform('count')


# <h1>preprocessing</h1>

# In[226]:


# We map our features into different types
categorical_features = ['os', 'browser', 'payment_currency', 'device', 'country', 'industry', 'utm_cluster_id',
                         'plan_id', 'avg_team_cat']
normalized_features = ['collection_21_days', 'mrr', 'created_trial_delta', 'created_subscription_delta',
                       'trial_subscription_delta', 'now_trial_delta', 'now_subscription_delta', 'now_created_delta', 
                       'now_churn_delta', 'churn_subscription_delta', 'company_size', 'survey_answers']
normalized_features = []
binary_features = ['survey_did_answer']
untouched_features = ['paying', 'is_subscription', 'is_churn']
KBinsDiscretized_features = []
target = ['lead_score']

# And create a column transformer to handle the manipulation for us
preprocess = make_column_transformer(
    (OneHotEncoder(), categorical_features),
    (Normalizer(), normalized_features),
    (Binarizer(), binary_features)
)


# In[227]:


if 'account_id' in accounts.columns:
    accounts.set_index('account_id', inplace=True)
    #test_accounts.set_index('account_id', inplace=True)

    # Filling empty values with default values 
def fill_empty_values(dataset):
    dataset.loc[:,categorical_features] = dataset[categorical_features].fillna('')
    dataset.loc[:,normalized_features + binary_features + untouched_features] = dataset[normalized_features + binary_features + untouched_features].fillna(0)
    return dataset

accounts = fill_empty_values(accounts)


# In[228]:


def under_sample(data, target, neg_pos_ratio=1):
    target_num = len(data[data[target] == 1])
    negative_idx = data[data[target] == 0].index
    positve_idx = data[data[target] == 1].index
    rnd_negative = np.random.choice(negative_idx , target_num * neg_pos_ratio, replace=False)
    under_sample_idx = np.concatenate([positve_idx, rnd_negative])
    return data.loc[under_sample_idx]


# In[229]:


# We fit our column transformer on both the train and the test sets
preprocess.fit(accounts.drop('lead_score',axis=1))

#dataset_train = accounts.loc[train_accounts.account_id]
dataset_train = under_sample(accounts.loc[train_accounts.account_id], 'lead_score', 10)
dataset_test = accounts.loc[test_accounts.account_id].drop('lead_score',axis=1)

# We use transform to finally manipulate the features of our training set
x = preprocess.transform(dataset_train.drop('lead_score',axis=1))
# Seperating the label
y = dataset_train['lead_score']
print(f'train data size {len(dataset_train)}')


# <h1>train model</h1>

# from sklearn.feature_selection import SelectFromModel
# 
# log_r = LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=-1)
# sfm = SelectFromModel(log_r, threshold=0.5)
# x = sfm.fit_transform(x, y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
# 
# #sm = SMOTE(random_state=42)
# #x_train, y_train = sm.fit_resample(x_train, y_train)
# 
# model = LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=-1) # 'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000]
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

scores = ['f1','recall'] # 'precision', 'recall', 'f1', 
hyparam_grid = [{
    'n_estimators': [5, 20, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5],
    'class_weight': ['balanced'],
}]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), hyparam_grid, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print()


# In[206]:


print(classification_report(y_test, y_pred, target_names=['not lead','lead']))
print('Acc:  {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('MCC: {}'.format(metrics.matthews_corrcoef(y_test, y_pred)))
print('F1:  {}'.format(metrics.f1_score(y_test, y_pred)))


# In[160]:


print(classification_report(y_test, y_pred, target_names=['not lead','lead']))
print('Acc:  {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('MCC: {}'.format(metrics.matthews_corrcoef(y_test, y_pred)))
print('F1:  {}'.format(metrics.f1_score(y_test, y_pred)))


# In[149]:


print(classification_report(y_test, y_pred, target_names=['not lead','lead']))
print('Acc:  {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('MCC: {}'.format(metrics.matthews_corrcoef(y_test, y_pred)))
print('F1:  {}'.format(metrics.f1_score(y_test, y_pred)))


# <h1>submit</h1>

# In[104]:


sfm.transform(x_submission)


# In[207]:


x_submission = preprocess.transform(dataset_test)
#x_submission = sfm.transform(x_submission)
y_pred_submission = model.predict(x_submission)
# Creating a dictionary where the keys are the account_ids
# and the values are your predictions
submission_account_ids = [str(int(i))for i in dataset_test.index]
predictions = dict(zip(submission_account_ids, map(int, y_pred_submission)))


# In[208]:


group_name = 'fRidaY'


# In[209]:


# We validate first that we actually send all the test accounts expected to be sent
if y_pred_submission.shape[0] != 71683 or len(submission_account_ids) != 71683:
  raise Exception("You have to send all of the accounts! Expected: (71683, 71683), Got: ({}, {})".format(y_pred_submission.shape[0], submission_account_ids.shape[0]))

if "group_name" not in vars() or group_name == "":
  group_name = input("Please enter your group's name:")

data = json.dumps({'submitter': group_name, 'predictions': predictions}).encode('utf-8')

req = request.Request("https://leaderboard.datahack.org.il/monday/api/",
                      headers={'Content-Type': 'application/json'},
                      data=data)

res = request.urlopen(req)
print(json.load(res))


# In[ ]:


scores = ['f1','recall'] # 'precision', 'recall', 'f1', 
hyparam_grid = [{
    'n_estimators': [5, 20, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5],
    'class_weight': ['balanced'],
}]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), hyparam_grid, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_res, y_res)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

