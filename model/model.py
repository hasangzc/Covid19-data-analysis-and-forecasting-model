import imp
import math
import statistics
from argparse import ArgumentParser
from ast import dump
from pathlib import Path
from pickle import dump, load
from re import T
from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential
from keras.optimizers import adam_v2
from prepocessing import DataPipeline
from sklearn.metrics import mean_squared_log_error


class KerasTrainer:
    """
    This class process the data, trains an Deep Learning Model and tests the trained model when necessary.
    """

    def __init__(self, args: ArgumentParser) -> NoReturn:
        """Init method.
        Args:
            args (ArgumentParser): The arguments of the training and testing session.
        Returns:
            NoReturn: This method does not return anything.
        """

        # Create the args object
        self._args = args

        # Process the data
        self._df = DataPipeline(pd.read_csv(f"./data/{args.data}.csv"), args=args)

        # Select the proper column and rows after 6 or 7 depending of the country:
        self._df = self._df[6:]

        # Create test and train datasets
        # Define the window of the RNN model and training set at 80%:
        self._window = 10
        self._train_len = math.ceil(len(self._df) * 0.8)
        self._train_data = self._df[0 : self._train_len]

        self._X_train = []
        self._Y_train = []

        for i in range(self._window, len(self._train_data)):
            self._X_train.append(self._train_data[i - self._window : i])
            self._Y_train.append(self._train_data[i])

        self._X_train, self._Y_train = np.array(self._X_train), np.array(self._Y_train)
        self._X_train = np.reshape(
            self._X_train, (self._X_train.shape[0], self._X_train.shape[1], 1)
        )

        # Define validation set at remaining 20%
        self._test_data = self._df[self._train_len - self._window :]
        self._X_val = []
        self._Y_val = []

        for i in range(self._window, len(self._test_data)):
            self._X_val.append(self._test_data[i - self._window : i])
            self._Y_val.append(self._test_data[i])

        self._X_val, self._Y_val = np.array(self._X_val), np.array(self._Y_val)
        self._X_val = np.reshape(
            self._X_val, (self._X_val.shape[0], self._X_val.shape[1], 1)
        )

    def _train(self) -> NoReturn:
        """This function trains an RNN model.

        Returns:
            NoReturn: This function does not return anything.
        """
        # Declare a saving path
        self.model_path = f"./saved_model/Keras/{self._args.data}"
        # Create the saving path if it does not exist
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        # Create a model
        self._model = Sequential()
        self._model.add(
            SimpleRNN(
                50,
                return_sequences=True,
                activation="relu",
                input_shape=(self._X_train.shape[1], 1),
            )
        )
        self._model.add(SimpleRNN(50, return_sequences=False, activation="relu"))
        self._model.add(Dense(100))
        self._model.add(Dense(25))
        self._model.add(Dense(1))

        self._opt = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        self._model.compile(loss="mean_squared_error", optimizer=self._opt)
        self._model.fit(
            self._X_train, self._Y_train, epochs=100, batch_size=10, verbose=1
        )

        # Save the model
        dump(
            obj=self._model, file=open(f"{self.model_path}/covid19_rnn_model.pkl", "wb")
        )

    def _test(self) -> NoReturn:
        """This function tests the test data using the last trained model.

        Returns:
            NoReturn: This function does not return anything.
        """
        self._train_pred = self._model.predict(self._X_train)
        self._val_pred = self._model.predict(self._X_val)

        # As we will compute logaritmic the argument must be greater or equal to 0
        self._train_pred[self._train_pred < 0] = 0
        self._val_pred[self._val_pred < 0] = 0

        self._msle_train = np.round(
            mean_squared_log_error(self._Y_train, self._train_pred), 3
        )
        self._msle_val = np.round(
            mean_squared_log_error(self._Y_val, self._val_pred), 3
        )

        print(f"Train data MSLE: {self._msle_train}  Test data MSLE: {self._msle_val}")

    def pipeline(self) -> NoReturn:
        # Train the model
        self._train()

        if self._args.visualize_predicts:
            # Make direction for result plot and dataframe
            Path(f"./visualization_results/predict_validation_results/").mkdir(
                parents=True, exist_ok=True
            )

            # Create new dataframe including actual validation data and predicted validation:
            valid = pd.DataFrame(self._df[self._train_len :])
            valid["Predictions"] = self._model.predict(self._X_val)

            # Save the dataframe
            valid.to_excel(
                f"./visualization_results/predict_validation_results/Predictions.xlsx",
                engine="xlsxwriter",
                index=False,
            )

            # Plot the both curves:
            plt.figure(figsize=(16, 8))
            plt.title(
                f"Validation and predicted values by RNN model in {self._args.location}"
            )
            plt.xlabel("Date")
            plt.ylabel("New cases")
            plt.plot(valid[["7_days_MA_new_cases", "Predictions"]])
            plt.legend(["Validation", "Predictions"])

            # Save the resulting plot
            plt.savefig(
                f"./visualization_results/predict_validation_results/predictions.png",
                bbox_inches="tight",
            )

        if self._args.is_testing:
            self._test()
