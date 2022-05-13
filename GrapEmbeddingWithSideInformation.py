# -*- coding: utf-8 -*-
from ge import Node2Vec
import networkx as nx
import json
import pandas as pd
from hashlib import md5


def clean_friends(row):
    if row == 'None':
        return []
    _friends = set(map(lambda _row: str(_row) + '_u', row.split(',')))
    return list(_friends.intersection(user_ids))


def category2word(row):
    if not row:
        return []
    if row == 'None':
        return []
    words = row.split(',')
    words = list(map(lambda _row: _row.replace('&', ' ').replace('/', ' ').replace('   ', ' ').strip(), words))
    words = [md5(word.encode(encoding='UTF-8')).hexdigest() for word in words if word is not None]
    return words


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
    data = pd.concat([train_set, eval_set, test_set])
elif mode == 'test':
    user_ids = sorted(list(set(eval_set.user_id.tolist() + train_set.user_id.tolist())))
    business_ids = sorted(list(set(eval_set.business_id.tolist() + train_set.business_id.tolist())))
    data = pd.concat([train_set, eval_set])
else:
    user_ids = sorted(list(set(train_set.user_id.tolist())))
    business_ids = sorted(list(set(train_set.business_id.tolist())))
    data = train_set

with open("../yelp_data/business.json", 'r', encoding='utf-8') as f:
    lines = map(json.loads, f.readlines())
business = pd.DataFrame.from_records(lines)
business.business_id = business.business_id.map(lambda row: str(row) + '_b')
business = pd.DataFrame(business[business.business_id.isin(business_ids)]).reset_index(drop=True)

with open("../yelp_data/user.json", 'r', encoding='utf-8') as f:
    lines = map(json.loads, f.readlines())
user = pd.DataFrame.from_records(lines)
user.user_id = user.user_id.map(lambda row: str(row) + '_u')
user = pd.DataFrame(user[user.user_id.isin(user_ids)]).reset_index(drop=True)

user['friend_lst'] = user.friends.map(clean_friends)

friend_pairs = set()
for _ in range(len(user)):
    user_id = user.user_id[_]
    friends = user.friend_lst[_]
    if not friends:
        continue
    for friend in friends:
        friend_pairs.add(tuple(sorted([user_id, friend])))

business.categories = business.categories.map(category2word)
categories = set()
for cats in business.categories.values:
    for cat in cats:
        categories.add(cat)
categories = sorted(list(categories))

business_pairs = set()
for _ in range(len(business)):
    business_id = business.business_id[_]
    cats = business.categories[_]
    if not cats:
        continue
    for cat in cats:
        business_pairs.add(tuple(sorted([business_id, cat])))

pairs = list(business_pairs.union(friend_pairs))

with open('../yelp_data/edges.txt', 'w+') as f:
    for line in data.values:
        f.write(f'{line[0]} {line[1]}\n')
    for pair in pairs:
        f.write(f'{pair[0]} {pair[1]}\n')

G = nx.read_edgelist('../yelp_data/edges.txt',
                     create_using=nx.DiGraph(),
                     nodetype=None,
                     data=[('weight', int)])

model = Node2Vec(G, walk_length=25, num_walks=250, p=0.25, q=4, workers=16)
model.train(embed_size=128, window_size=15, iter=40, workers=16)  # train model
embeddings = model.get_embeddings()

df_embeddings = pd.DataFrame.from_dict(embeddings, orient='index',
                                       columns=[f'E_{_}' for _ in range(128)]).reset_index().rename(
    columns={'index': 'id'})
df_embeddings.to_csv('../yelp_data/embeddings.csv', index=False)
