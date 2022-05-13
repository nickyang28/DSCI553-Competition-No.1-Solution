# -*- coding:utf-8 -*-
# Model Description
# The way I improved RMSE is through Graph Embedding. See the file GraphEmbedding.
# In the graph, we have users, business and categories. So we have embeddings of them.
# For each business, we use the average embedding of its categories.
# We use node2vec to generate embeddings;
# So the main model contains 4 parts of information:
# 1. User Embedding;
# 2. Business Embedding:
# 3. Category Embedding;
# 4. Other features in the dataset.
# In the val set, there are some business that are not seen before, so we train a different model that does not contain business embedding.
# So the user side model contains 3 parts of information
# 1. User Embedding;
# 2. Category Embedding;
# 3. Other features in the dataset.
# To prevent from error. we also trained a plain model that only has the features in the dataset.

# Also, to improve the model, we used model stacking: 10-fold xgbt + 10-fold catboost.
# Train on the train test and evaluation on the val set:
# Both: 0.9724
# User: 0.9735
# Combined: 0.9726

# Since the final score is on test set. So we trained the model on both train and val set.
# Therefore, the score .9043 on leaderboard is not real.
# My estimation of the RMSE on the test set will be around 0.9695.
# Running time will be around 300s.

import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import json
import pickle
from catboost import Pool, CatBoostRegressor
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path")
    parser.add_argument("test_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    with open(args.folder_path + "user.json", 'r', encoding='utf-8') as f:
        lines = map(json.loads, f.readlines())
        user = pd.DataFrame.from_records(lines)

    user.user_id = user.user_id.map(lambda row: str(row) + '_u')
    user['review_count_user'] = user.review_count
    user['yelping_since'] = user.yelping_since.map(lambda row: int(row[:4]))
    user_dense = ['review_count_user', 'useful', 'funny', 'cool', 'fans', 'average_stars',
                  'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                  'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                  'compliment_funny', 'compliment_writer', 'compliment_photos']
    user_features = user[['user_id', 'yelping_since'] + user_dense]
    del user
    print('Done User.')

    business_features = pd.read_csv('business.csv').fillna(0)
    print('Done Business.')

    embeddings = pd.read_csv('embeddings.csv')
    print('Done Embedding.')

    temp = pd.read_csv(args.test_file)
    test_set = pd.DataFrame()
    test_set['user_id'] = temp.iloc[:, 0].map(lambda row: str(row) + '_u')
    test_set['business_id'] = temp.iloc[:, 1].map(lambda row: str(row) + '_b')

    both_lst = []
    user_lst = []
    plain_lst = []
    bus_avg = []
    usr_avg = []
    avg = []

    eids = embeddings.id.tolist()
    bids = business_features.business_id.tolist()
    uids = user_features.user_id.tolist()

    for _ in tqdm.tqdm(range(len(test_set))):
        bid = test_set.business_id[_]
        uid = test_set.user_id[_]
        if (uid in eids) and (bid in eids):
            both_lst.append(_)
        elif (uid in eids) and (bid not in eids) and (bid in bids):
            user_lst.append(_)
        elif (uid in eids) and (bid not in bids):
            usr_avg.append(_)
        elif (uid not in eids) and (uid in uids) and (bid in bids):
            plain_lst.append(_)
        elif (uid not in uids) and (bid in bids):
            bus_avg.append(_)
        else:
            avg.append(_)

    both_set = pd.DataFrame(test_set.iloc[both_lst, :])
    user_set = pd.DataFrame(test_set.iloc[user_lst, :])
    plain_set = pd.DataFrame(test_set.iloc[plain_lst, :])
    bus_avg_set = pd.DataFrame(test_set.iloc[bus_avg, :])
    usr_avg_set = pd.DataFrame(test_set.iloc[usr_avg, :])
    avg_set = pd.DataFrame(test_set.iloc[avg, :])

    # begain both
    X_both = both_set \
        .merge(embeddings, left_on='business_id', right_on='id', how='left') \
        .merge(embeddings, left_on='user_id', right_on='id', how='left', suffixes=('_business', '_user')) \
        .merge(business_features, on='business_id', how='left') \
        .merge(user_features, on='user_id', how='left') \
        .drop(['user_id', 'business_id', 'id_business', 'id_user'], axis=1) \
        .fillna(0)

    res = pd.DataFrame()
    cat_features = [X_both.columns.tolist().index(cat) for cat in ['state', 'yelping_since']]
    test_pool = Pool(X_both, cat_features=cat_features)
    for _ in range(10):
        model = pickle.load(open(f'./models/both/model_{_}_xgbt.pkl', 'rb'))
        res[f'res_{_}'] = list(model.predict(X_both))
        model = pickle.load(open(f'./models/both/model_{_}_cat.pkl', 'rb'))
        res[f'res_{_}_'] = list(model.predict(test_pool))

    pred = list(np.mean(res.values, axis=1))
    both_set['prediction'] = pred

    # begain user
    X_user = user_set \
        .merge(embeddings, left_on='user_id', right_on='id', how='left', suffixes=('_business', '_user')) \
        .merge(business_features, on='business_id', how='left') \
        .merge(user_features, on='user_id', how='left') \
        .drop(['user_id', 'business_id', 'id'], axis=1) \
        .fillna(0)

    res = pd.DataFrame()
    cat_features = [X_user.columns.tolist().index(cat) for cat in ['state', 'yelping_since']]
    test_pool = Pool(X_user, cat_features=cat_features)
    for _ in range(10):
        model = pickle.load(open(f'./models/user/model_{_}_xgbt.pkl', 'rb'))
        res[f'res_{_}'] = list(model.predict(X_user))
        model = pickle.load(open(f'./models/user/model_{_}_cat.pkl', 'rb'))
        res[f'res_{_}_'] = list(model.predict(test_pool))

    pred = list(np.mean(res.values, axis=1))
    user_set['prediction'] = pred

    # begain plain
    X_plain = plain_set \
        .merge(business_features, on='business_id', how='left') \
        .merge(user_features, on='user_id', how='left') \
        .drop(['user_id', 'business_id'], axis=1) \
        .fillna(0)

    res = pd.DataFrame()

    for _ in range(10):
        model = pickle.load(open(f'./models/plain/model_{_}_xgbt.pkl', 'rb'))
        res[f'res_{_}'] = list(model.predict(X_plain))

    pred = list(np.mean(res.values, axis=1))
    plain_set['prediction'] = pred

    # begain bus_avg_set
    bus_avg_set['prediction'] = 0
    for idx, bid in bus_avg_set.business_id.items():
        bif = business_features[business_features.business_id == bid].stars.values[0]
        bus_avg_set.loc[idx, 'prediction'] = bif

    # begain usr_avg_set
    usr_avg_set['prediction'] = 0
    for idx, uid in usr_avg_set.user_id.items():
        uif = user_features[user_features.user_id == uid].average_stars.values[0]
        usr_avg_set.loc[idx, 'prediction'] = uif

    # begain avg_set
    avg_set['prediction'] = 3.7510110420172

    result = pd.concat([both_set, user_set, plain_set, bus_avg_set, usr_avg_set, avg_set]).sort_index()
    result.user_id = result.user_id.map(lambda row: str(row)[:-2])
    result.business_id = result.business_id.map(lambda row: str(row)[:-2])
    result.to_csv(args.out_file, index=False)
