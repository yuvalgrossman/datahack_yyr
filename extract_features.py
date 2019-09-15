import modin.pandas as pd
from time import time

data_path = './data/'
tic=time()
train_accounts = pd.read_csv(data_path + 'train_accounts.csv')#.sample(frac=0.2)
train_users = pd.read_csv(data_path + 'train_users.csv')
train_events = pd.read_csv(data_path + 'train_events.csv')
train_subscriptions = pd.read_csv(data_path + 'train_subscriptions.csv')
test_accounts = pd.read_csv(data_path + 'test_accounts.csv')
test_users = pd.read_csv(data_path + 'test_users.csv')
test_events = pd.read_csv(data_path + 'test_events.csv')
test_subscriptions = pd.read_csv(data_path + 'test_subscriptions.csv')
print('import time: {}'.format(time()-tic))
tic=time()
accounts = pd.concat([train_accounts, test_accounts],sort=False)
print(f'accounts: {len(accounts)}')
users = pd.concat([train_users, test_users],sort=False)
print(f'users: {len(users)}')
events = pd.concat([train_events, test_events],sort=False)
print(f'events: {len(events)}')
subscriptions = pd.concat([train_subscriptions, test_subscriptions],sort=False)
print(f'subscriptions: {len(subscriptions)}')
print('concat time: {}'.format(time()-tic))

def extract_num_feat(df,d):
    cols=df.select_dtypes('number').columns # columns names
    n_df1 = accounts.set_index('account_id').loc[:,['lead_score']] #new dataframe with account as index and columns lead_score
    n_df2 = df.groupby('account_id').filter(lambda x: x[df.columns[1]].count()>d).groupby('account_id')[cols].describe()
    n_df = n_df2.join(n_df1)
    return n_df

num=500
tic=time()
df = extract_num_feat(events,num)
print('calculation time for events: {}'.format(time()-tic))
df.to_csv('accounts_above+{}+events_feat.csv'.format(num))

tic=time()
df = extract_num_feat(users,num)
print('calculation time for users: {}'.format(time()-tic))
df.to_csv('accounts_above+{}+users_feat.csv'.format(num))

tic=time()
df = extract_num_feat(subscriptions,num)
print('calculation time for subscriptions: {}'.format(time()-tic))
df.to_csv('accounts_above+{}+subscriptions_feat.csv'.format(num))