# coding: utf-8
# The main model contains 4 parts of information:
# 1. User Embedding;
# 2. Business Embedding:
# 3. Category Embedding;
# 4. Other features in the dataset.
import pandas as pd
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
from hashlib import md5
from catboost import Pool, CatBoostRegressor


def category2word(row):
    if not row:
        return []
    if row == 'None':
        return []
    words = row.split(',')
    words = list(map(lambda _row: _row.replace('&', ' ').replace('/', ' ').replace('   ', ' ').strip(), words))
    words = [md5(word.encode(encoding='UTF-8')).hexdigest() for word in words if word is not None]
    return words


def cat_emb(row):
    df = pd.DataFrame()
    df['id'] = row
    df = df.merge(embeddings, on='id', how='left').iloc[:, 1:]
    df = df.mean(axis=0).tolist()
    return df


embeddings = pd.read_csv('../yelp_data/embeddings.csv')

mode = 'online'  # mode in ['online', 'test'. 'train']
train_set = pd.read_csv('../yelp_data/yelp_train.csv').rename(columns={'stars': 'target'})
eval_set = pd.read_csv('../yelp_data/yelp_val.csv').rename(columns={'stars': 'target'})
test_set = pd.read_csv('../yelp_data/yelp_test_ans.csv').rename(columns={'stars': 'target'})

train_set.user_id = train_set.user_id.map(lambda row: str(row) + '_u')
train_set.business_id = train_set.business_id.map(lambda row: str(row) + '_b')
eval_set.user_id = eval_set.user_id.map(lambda row: str(row) + '_u')
eval_set.business_id = eval_set.business_id.map(lambda row: str(row) + '_b')
test_set.user_id = test_set.user_id.map(lambda row: str(row) + '_u')
test_set.business_id = test_set.business_id.map(lambda row: str(row) + '_b')

if mode == 'online':
    user_ids = sorted(list(set(eval_set.user_id.tolist() + train_set.user_id.tolist() + test_set.user_id.tolist())))
    business_ids = sorted(
        list(set(eval_set.business_id.tolist() + train_set.business_id.tolist() + test_set.business_id.tolist())))
elif mode == 'test':
    user_ids = sorted(list(set(eval_set.user_id.tolist() + train_set.user_id.tolist())))
    business_ids = sorted(list(set(eval_set.business_id.tolist() + train_set.business_id.tolist())))
else:
    user_ids = sorted(list(set(train_set.user_id.tolist())))
    business_ids = sorted(list(set(train_set.business_id.tolist())))

with open("../yelp_data/business.json", 'r', encoding='utf-8') as f:
    lines = map(json.loads, f.readlines())
business = pd.DataFrame.from_records(lines)
business.business_id = business.business_id.map(lambda row: str(row) + '_b')
business.categories = business.categories.map(category2word)

states = sorted(list(business.state.unique()))
sta2idx = {sta: _ for _, sta in enumerate(states)}
business.state = business.state.map(lambda row: sta2idx[row])

post_codes = sorted(list(business.postal_code.unique()))
pos2idx = {pos: _ for _, pos in enumerate(post_codes)}
business.postal_code = business.postal_code.map(lambda row: pos2idx[row])

business['review_count_business'] = business.review_count
business_features = pd.DataFrame(
    business[['business_id', 'state', 'is_open', 'stars', 'review_count_business', 'categories']])
del business

temp = business_features.apply(lambda row: cat_emb(row.categories), axis='columns', result_type='expand').rename(
    columns={_: f'E_{_}_category' for _ in range(128)})
business_features.drop(columns=['categories'], inplace=True)
business_features = pd.concat([business_features, temp], axis='columns').fillna(0)
business_features.to_csv('business.csv', index=False)
business_features = pd.read_csv('business.csv').fillna(0)

with open("../yelp_data/user.json", 'r', encoding='utf-8') as f:
    lines = map(json.loads, f.readlines())
user = pd.DataFrame.from_records(lines)
user.user_id = user.user_id.map(lambda row: str(row) + '_u')

user['review_count_user'] = user.review_count
user['yelping_since'] = user.yelping_since.map(lambda row: int(row[:4]))

user_dense = ['review_count_user', 'useful', 'funny', 'cool', 'fans', 'average_stars',
              'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
              'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
              'compliment_funny', 'compliment_writer', 'compliment_photos']
for feature in user_dense:
    if feature == 'average_stars':
        continue

user_features = user[['user_id', 'yelping_since'] + user_dense]
del user

train_data = train_set.merge(embeddings, left_on='business_id', right_on='id', how='left').merge(embeddings, left_on='user_id', right_on='id', how='left', suffixes=('_business', '_user')).merge(business_features, on='business_id', how='left').merge(user_features, on='user_id', how='left').drop(['user_id', 'business_id', 'id_business', 'id_user'], axis=1).fillna(0)
eval_data = eval_set.merge(embeddings, left_on='business_id', right_on='id', how='left').merge(embeddings, left_on='user_id', right_on='id', how='left', suffixes=('_business', '_user')).merge(business_features, on='business_id', how='left').merge(user_features, on='user_id', how='left').drop(['user_id', 'business_id', 'id_business', 'id_user'], axis=1).fillna(0)
test_data = test_set.merge(embeddings, left_on='business_id', right_on='id', how='left').merge(embeddings, left_on='user_id', right_on='id', how='left', suffixes=('_business', '_user')).merge(business_features, on='business_id', how='left').merge(user_features, on='user_id', how='left').drop(['user_id', 'business_id', 'id_business', 'id_user'], axis=1).fillna(0)

X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
X_eval, y_eval = eval_data.iloc[:, 1:], eval_data.iloc[:, 0]
X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

if mode == 'test':
    X = pd.concat([X_train, X_eval])
    y = pd.concat([y_train, y_eval])
elif mode == 'train':
    X = X_train
    y = y_train
else:
    X = pd.concat([X_train, X_eval, X_test])
    y = pd.concat([y_train, y_eval, y_test])

cv = KFold(n_splits=10, random_state=553, shuffle=True)
for i, (train_index, test_index) in enumerate(cv.split(X)):
    print("-" * 10 + str(i) + "-" * 10)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = xgb.XGBRegressor(n_estimators=512, max_depth=6, learning_rate=0.1, tree_method='gpu_hist')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_pred, y_train)
    print("Train RMSE: %.4f" % (mse ** 0.5))

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    print("Eval RMSE: %.4f" % (mse ** 0.5))

    pickle.dump(model, open(f"./models/both/model_{i}_xgbt.pkl", 'wb'))

cat_features = [X.columns.tolist().index(cat) for cat in ['state', 'yelping_since']]
for i, (train_index, test_index) in enumerate(cv.split(X)):
    print("-" * 10 + str(i) + "-" * 10)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)

    model = CatBoostRegressor(n_estimators=512, depth=10, learning_rate=0.1, loss_function='RMSE', random_state=553,
                              silent=True, task_type="GPU", devices='0')

    model.fit(train_pool)
    y_pred = model.predict(train_pool)
    mse = mean_squared_error(y_pred, y_train)
    print("Train RMSE: %.4f" % (mse ** 0.5))

    y_pred = model.predict(test_pool)
    mse = mean_squared_error(y_pred, y_test)
    print("Eval RMSE: %.4f" % (mse ** 0.5))

    pickle.dump(model, open(f"./models/both/model_{i}_cat.pkl", 'wb'))
