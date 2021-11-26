import math
import pickle
from functools import partial

import pandas as pd
import torch
import torchmetrics
from accelerate import notebook_launcher
from torch import nn
from torch.utils.data import Dataset

from pytorch_accelerated import Trainer, TrainerPlaceholderValues
from pytorch_accelerated.callbacks import SaveBestModelCallback
from pytorch_accelerated.trainer import DEFAULT_CALLBACKS


class Embedding(nn.Embedding):
    """Embedding layer with truncated normal initialization, see https://arxiv.org/abs/1711.09160"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        trunc_normal_(self.weight.data, std=0.01)


def trunc_normal_(x, mean=0.0, std=1.0):
    """
    Truncated normal initialization (approximation)
    From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    """
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class MfDotBias(torch.nn.Module):
    """General Matrix Factorization model"""

    def __init__(
            self, n_factors, n_users, n_items, ratings_range=None, use_biases=True
    ):
        super().__init__()
        self.bias = use_biases
        self.y_range = ratings_range
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        if use_biases:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)

    def forward(self, inputs):
        users, items = inputs
        dot = self.user_embedding(users) * self.item_embedding(items)
        res = dot.sum(1)
        if self.bias:
            res = (
                    res + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
            )

        if self.y_range is None:
            return res
        else:
            return (
                    torch.sigmoid(res) * (self.y_range[1] - self.y_range[0])
                    + self.y_range[0]
            )


class UserItemRatings(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __getitem__(self, index):
        row = self.df.iloc[index]
        user_id = self.user_lookup[row.user_id]
        movie_id = self.movie_lookup[row.title]

        rating = torch.tensor(row.rating, dtype=torch.float32)

        return (user_id, movie_id), rating

    def __len__(self):
        return len(self.df)


class BstTransformerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def calculate_train_batch_loss(self, batch) -> dict:
        inputs, targets = batch
        out = self.model(inputs)

        loss = self.loss_func(out, targets)

        return {
            'loss': loss,
            'model_outputs': out,
            'batch_size': batch[1].size(0)
        }

    def calculate_eval_batch_loss(self, batch) -> dict:
        with torch.no_grad():
            inputs, targets = batch
            out = self.model(inputs)
            loss = self.loss_func(out, targets)

        self.mae.update(out, targets)
        self.mse.update(out, targets)

        return {
            'loss': loss,
            'model_outputs': out,
            'batch_size': batch[1].size(0)
        }

    def eval_epoch_end(self):
        self.run_history.update_metric(
            "mae", self.mae.compute().cpu()
        )
        mse = self.mse.compute().item()
        self.run_history.update_metric("mse", mse)
        self.run_history.update_metric('rmse', math.sqrt(mse))

        self.mae.reset()
        self.mse.reset()

def main():
    ratings_df = pd.read_csv('ratings.csv')

    user_lookup = {v: i + 1 for i, v in enumerate(ratings_df['user_id'].unique())}

    with open('movie_lookup.pickle', 'rb') as f:
        movie_lookup = pickle.load(f)

    train_dataset = UserItemRatings(ratings_df[ratings_df.is_valid == False], movie_lookup, user_lookup)
    valid_dataset = UserItemRatings(ratings_df[ratings_df.is_valid == True], movie_lookup, user_lookup)

    model = MfDotBias(120, len(user_lookup), len(movie_lookup), ratings_range=[0.5, 5.5])
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    create_sched_fn = partial(
        torch.optim.lr_scheduler.OneCycleLR,
        max_lr=0.01,
        epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
    )

    trainer = BstTransformerTrainer(model=model, loss_func=loss_func, optimizer=optimizer,
                                    callbacks=(*DEFAULT_CALLBACKS, SaveBestModelCallback(watch_metric='mae')))

    trainer.train(train_dataset=train_dataset,
                  eval_dataset=valid_dataset,
                  num_epochs=10,
                  per_device_batch_size=512,
                  create_scheduler_fn=create_sched_fn,
                  )

if __name__ == '__main__':
    notebook_launcher(main, num_processes=2)