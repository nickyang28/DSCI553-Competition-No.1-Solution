# DSCI553-Competition-No.1-Solution
USC ✌️ 2022 Spring DSCI 553 (Foundations and Applications of Data Mining) | Recommendation System Competition | 1st Solution

**We Will Not Include Any Source Data Here.**

We don't know the final RMSE on the hidden test set, but my guess is around 0.9670.

## First Step: Train Graph Embedding

Run [GrapEmbeddingWithSideInformation.py](GrapEmbeddingWithSideInformation.py) to obtain the embedding for User, Business, and Category. It takes around 30 minutes.

```python
model = Node2Vec(G, walk_length=25, num_walks=250, p=0.25, q=4, workers=16)
model.train(embed_size=128, window_size=15, iter=40, workers=16)
embeddings = model.get_embeddings()
```

## Second Step: Train Model With 4 Parts of Information

```markdown
1. User Embedding;
2. Business Embedding:
3. Category Embedding;
4. Other features in the dataset.
```

Run  [TrainBothModel.py](TrainBothModel.py) to train. If you have a GPU, it may take you 30 minutes.

## Third Step: Train User Side Model

In the validation set, there are some business that are not seen before, so we train a different model that does not contain business embeddings.

```markdown
1. User Embedding;
2. Category Embedding;
3. Other features in the dataset.
```

Run  [TrainUserModel.py](TrainUserModel.py) to train. If you have a GPU, it may take you 25 minutes.

## Fourth Step: Train Plain Side Model

To prevent from error, we also trained a plain model that only has the features in the dataset. But in practice, this model is never called.

Run  [TrainPlainModel.py](TrainPlainModel.py) to train. If you have a GPU, it may take you 20 minutes.

## Last Step: Online Test

Use [Competition.py](Competition.py) to upload online. To improve the model, we used model stacking: 10-fold xgbt + 10-fold CatBoost.

