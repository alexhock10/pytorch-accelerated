import math

import pandas as pd
import torch
import torch.utils.data as data
import torchmetrics
from torch import nn

from pytorch_accelerated.trainer import Trainer


class MovieDataset(data.Dataset):

    def __init__(
            self, ratings_file, test=False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with user,past,future.
        """
        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        movie_history = eval(data.sequence_movie_ids)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]
        target_movie_rating = movie_history_ratings[-1:][0]

        movie_history = torch.LongTensor(movie_history[:-1])
        movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])

        sex = data.sex
        age_group = data.age_group
        occupation = data.occupation

        return user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation


class BstTransformer(nn.Module):

    def __init__(self, users, movies, sequence_length, genres):
        super().__init__()
        self.sequence_length = sequence_length
        self.register_buffer('dummy_param', torch.empty(0))

        self.embeddings_user_id = nn.Embedding(
            int(users.user_id.max()) + 1, int(math.sqrt(users.user_id.max())) + 1
        )
        ###Users features embeddings
        self.embeddings_user_sex = nn.Embedding(
            len(users.sex.unique()), int(math.sqrt(len(users.sex.unique())))
        )
        self.embeddings_age_group = nn.Embedding(
            len(users.age_group.unique()), int(math.sqrt(len(users.age_group.unique())))
        )
        self.embeddings_user_occupation = nn.Embedding(
            len(users.occupation.unique()), int(math.sqrt(len(users.occupation.unique())))
        )
        # self.embeddings_user_zip_code = nn.Embedding(
        #     len(users.zip_code.unique()), int(math.sqrt(len(users.sex.unique())))
        # )

        ##Movies
        self.embeddings_movie_id = nn.Embedding(
            int(movies.movie_id.max()) + 1, int(math.sqrt(movies.movie_id.max())) + 1
        )
        self.embeddings_position = nn.Embedding(
            sequence_length, int(math.sqrt(len(movies.movie_id.unique()))) + 1
        )
        ###Movies features embeddings
        genre_vectors = movies[genres].to_numpy()
        self.embeddings_movie_genre = nn.Embedding(
            genre_vectors.shape[0], genre_vectors.shape[1]
        )

        self.embeddings_movie_genre.weight.requires_grad = False  # Not training genres

        # self.embeddings_movie_year = nn.Embedding(
        #     len(movies.year.unique()), int(math.sqrt(len(movies.year.unique())))
        # )

        self.transformerlayer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=63, nhead=3, dropout=0.2),
            num_layers=1
        )

        self.linear = nn.Sequential(
            nn.Linear(
                589,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def encode_input(self, inputs):
        user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation = inputs

        # MOVIES
        movie_history = self.embeddings_movie_id(movie_history)
        target_movie = self.embeddings_movie_id(target_movie_id)

        positions = torch.arange(0, self.sequence_length - 1, 1, dtype=int, device=self.dummy_param.device)
        positions = self.embeddings_position(positions)

        encoded_sequence_movies_with_position_and_rating = (movie_history + positions)  # Yet to multiply by rating

        target_movie = torch.unsqueeze(target_movie, 1)
        transformer_features = torch.cat((encoded_sequence_movies_with_position_and_rating, target_movie), dim=1)

        # USERS
        user_id = self.embeddings_user_id(user_id)

        sex = self.embeddings_user_sex(sex)
        age_group = self.embeddings_age_group(age_group)
        occupation = self.embeddings_user_occupation(occupation)
        user_features = torch.cat((user_id, sex, age_group, occupation), 1)

        return transformer_features, user_features, target_movie_rating.float()

    def forward(self, batch):
        transformer_features, user_features, target_movie_rating = self.encode_input(batch)
        transformer_output = self.transformerlayer(transformer_features)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        # Concat with other features
        features = torch.cat((transformer_output, user_features), dim=1)

        output = self.linear(features)
        return output, target_movie_rating


class BstTransformerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def training_run_start(self):
        self.mae.to(self._eval_dataloader.device)
        self.mse.to(self._eval_dataloader.device)

    def calculate_train_batch_loss(self, batch) -> dict:
        out, target_movie_rating = self.model(batch)
        loss = self.loss_func(out.flatten(), target_movie_rating)

        return {
            'loss': loss,
            'model_outputs': out,
            'batch_size': batch[0].size(0)
        }

    def calculate_eval_batch_loss(self, batch) -> dict:
        with torch.no_grad():
            out, target_movie_rating = self.model(batch)
            out = out.flatten()
            loss = self.loss_func(out, target_movie_rating)

        self.mae.update(out, target_movie_rating)
        self.mse.update(out, target_movie_rating)

        return {
            'loss': loss,
            'model_outputs': out,
            'batch_size': batch[0].size(0)
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
    users = pd.read_csv(
        "/home/chris/notebooks/recs/data/users.csv",
        sep=",",
    )

    ratings = pd.read_csv(
        "/home/chris/notebooks/recs/data/ratings.csv",
        sep=",",
    )

    movies = pd.read_csv(
        "/home/chris/notebooks/recs/data/movies.csv", sep=","
    )

    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    for genre in genres:
        movies[genre] = movies["genres"].apply(
            lambda values: int(genre in values.split("|"))
        )

    sequence_length = 8

    train_dataset = MovieDataset("/home/chris/notebooks/recs/data/train_data.csv")
    val_dataset = MovieDataset("/home/chris/notebooks/recs/data/test_data.csv")

    model = BstTransformer(users=users,
                           movies=movies, genres=genres,
                           sequence_length=sequence_length)

    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    trainer = BstTransformerTrainer(model=model, loss_func=loss_func, optimizer=optimizer)

    trainer.train(train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  num_epochs=50,
                  per_device_batch_size=256,
                  # train_dataloader_kwargs={'num_workers': 0,
                  #                          'pin_memory': False}
                  gradient_accumulation_steps=2
                  )


if __name__ == '__main__':
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
    # notebook_launcher(main, num_processes=1)
    main()
