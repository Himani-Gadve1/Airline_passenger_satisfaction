# In[]:
from typing import Dict, List, Tuple  # , Union, Callable
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

train_path = "data/air_training.csv"
test_path = None

# The best columns were selected manually through an EDA,
# please see "best_columns.ipynb" for details.
best_columns = [
    "Online.boarding",
    "Inflight.wifi.service",
    "Type.of.Travel",
    "Inflight.entertainment",
    "Customer.Type",
    "Seat.comfort",
    "Checkin.service",
    "Age",
    "On.board.service",
    "Flight.Distance",
    "Inflight.service",
    "Baggage.handling",
    "Cleanliness",
    "Gate.location",
    "Arrival.Delay.in.Minutes",
]


# In[]:
class SVC_(SVC):
    def __init__(self):
        super().__init__()

    def finetune_(self, X, y, param_dict):
        """Your code here"""
        self.best_params = {}
        pass

    def fit_(self, X, y):
        """Your code here"""
        self.fit(X, y)
        pass

    def predict_(self, X):
        """Your code here"""
        return self.predict(X)


class NB_(GaussianNB):
    def __init__(self):
        """Your code here"""
        super().__init__()

    def finetune_(self, X, y, param_dict):
        """Your code here"""
        self.best_params = {}
        pass

    def fit_(self, X, y):
        """Your code here"""
        self.fit(X, y)
        pass

    def predict_(self, X):
        """Your code here"""
        return self.predict(X)


class NeuralNet:
    def __init__(self, mu: float = 1.0, sigma: float = 1.0):
        """Your code here"""
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        pass

    def fit():
        """Your code here"""
        pass

    def get_embeddings():
        """Your code here"""
        pass

    def predict():
        """Your code here"""
        pass


class MetaModel:
    def __init__(self):
        """Your code here"""
        pass

    def fit(self):
        """Your code here"""
        pass

    def predict(self):
        """Your code here"""
        pass


class Solution:
    def __init__(
        self,
        best_columns: List = best_columns,
        train_path: str = train_path,
        test_path: str = test_path,
        random_state: int = 42,
        dep_var: str = "satisfaction",
    ) -> Dict:
        """Train and finetune all models. Predict test set if provided.

        Args:
            best_columns (List, optional): Manually selected best columns.
            train_path (str, optional): Path to the training set.
            test_path (str, optional): PAth to the test set.
        Returns:
            Dict : validation or test predictions from 4 models
        """

        self.random_state = random_state
        self.dep_var = dep_var

        self.train_df = pd.read_csv(train_path)
        self.train_df = self._preproc(self.train_df)
        if test_path is not None:
            self.test_df = pd.read_csv(test_path)
            self.test_df = self._preproc(self.test_df)

        self.val_idx, self.test_idx = self._split(self.train_df)

        # train NN:
        NN = NeuralNet()
        NN.fit(self.train_df, self.val_idx, self.test_idx)
        train_df_emb = NN.get_embeddings(self.train_df)
        self.val_preds["NN"] = NN.predict(self.train_df.loc[self.val_idx, :])

        # train SVC & NB on embeddings:
        X_train = train_df_emb.loc[self.train_idx, :]
        X_val = train_df_emb.loc[self.val_idx, :]
        y_train = self.train_df.loc[self.train_idx, self.dep_var]
        y_val = self.train_df.loc[self.val_idx, self.dep_var]

        svc = SVC_()
        svc.fit(X_train, X_val, y_train, y_val, finetune=True)
        self.val_preds["SVC"] = svc.predict(X_val)

        nvb = NB_()
        nvb.fit(X_train, X_val, y_train, y_val, finetune=True)
        self.val_preds["NVB"] = nvb.predict(X_val)

        # train stacking on validation predictions:
        mm = MetaModel()
        mm.fit(self.val_preds, y_val)

        # predict test
        # use models to predict test:
        if test_path is not None:
            self.test_preds["NN"] = NN.predict(self.test_df)
            test_df_emb = NN.get_embeddings(self.test_df)
            self.test_preds["SVC"] = svc.predict(test_df_emb)
            self.test_preds["NVB"] = nvb.predict(test_df_emb)
            self.test_preds["MM"] = mm.predict(self.test_preds)
            return self.test_preds
        else:
            return self.val_preds

    def _preproc(self, df_: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Cast numeric columns to float32 and
        object columns to numeric categories, deletes ID.

        Args:
            df_ (pd.DataFrame): train or test dataset

        Returns:
            pd.DataFrame: cleaned copy of the dataset.
        """
        if inplace:
            df = df_.copy()

        # Take care of the target:
        if self.dep_var in df.columns:
            df.satisfaction = (df.satisfaction == "satisfied") * 1

        # Delete ID as it is useless unless it has a leak in it.
        del df["id"]

        # From EDA we can see that 0 is extremely rare value.
        # so we can probably cast it to NA.
        lowest, highest = 1, 5
        cat_list = list(range(lowest, highest + 1))
        cat_clmns = [
            clmn for clmn in df.columns if df[clmn].unique().shape[0] in {5, 6}
        ]
        for clmn in cat_clmns:
            df.loc[:, clmn] = pd.Categorical(
                df[clmn], categories=cat_list, ordered=True
            )

        # Class seems to be an ordered category:
        class_order = ["Eco", "Eco Plus", "Business"]
        df.Class = pd.Categorical(df.Class, categories=class_order, ordered=True)
        df.Class = df.Class.cat.codes.astype("category")

        # cast object columns to numeric categories:
        for obj_clmn in df.columns[df.dtypes == "object"]:
            df.loc[:, obj_clmn] = df[obj_clmn].astype("category")
            df.loc[:, obj_clmn] = df[obj_clmn].cat.codes.astype("category")

        # cast all numerical to float32 to prevent errors in TabularLearner:
        for clmn in df.columns[df.dtypes != "category"]:
            df.loc[:, clmn] = df[clmn].astype("category").astype(np.float32)

        return df

    def _split(
        self,
        train_df: pd.DataFrame,
        strata_columns: List[str] = ["Class", "Type.of.Travel"],
        val_size: float = 0.3,
    ) -> Tuple(np.Array, np.Array):
        """Split dataset into train and validation.

        Args:
            train_df (pd.DataFrame): train dataset to split
            strata_columns (List[str], optional):
                list of columns to use for stratification.
                Defaults to ['Class','Type.of.Travel'].
            val_szie (float, optional):
                Size of validation set. Defaults to 0.3.

        Returns:
            Tuple(np.Array, np.Array):
                train and validation indices as np.arrays.
        """
        strata = pd.Series("starta_", index=train_df.index)
        for clmn in strata_columns:
            strata = strata + train_df[clmn].astype(str)

        X_train, X_test, _, _ = train_test_split(
            train_df,
            train_df[self.dep_var],
            test_size=val_size,
            random_state=self.random_state,
            stratify=strata,
        )

        return X_train.index.values, X_test.index.values


# %%
