from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from model.model import KerasTrainer


def declareParserArguments(parser: ArgumentParser) -> ArgumentParser:
    # Add arguments
    parser.add_argument(
        "--location", default="Turkey", type=str, help="Country to be analyzed."
    )

    parser.add_argument(
        "--informations",
        action="store_true",
        default=False,
        help="Information about filtered data.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="owid-covid-data",
        help="The data file name [without .csv]",
    )

    parser.add_argument(
        "--is_testing",
        action="store_true",
        default=False,
        help="Whether to test the trained model.",
    )

    parser.add_argument(
        "--visualize_predicts",
        action="store_true",
        default=False,
        help="Whether to visualize the test results",
    )

    return parser.parse_args()


# If this script run directly
if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Covid19 Forecasting")
    # Add the arguments
    args = declareParserArguments(parser=parser)

    # Create a KerasTrainer object
    keras_trainer = KerasTrainer(args=args)
    # Initiate the training & testing pipeline
    keras_trainer.pipeline()
