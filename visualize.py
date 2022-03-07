from argparse import ArgumentParser
from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import validation_curve
from model.model import KerasTrainer

from model import model
from prepocessing import DataPipeline, feature_engineering, normalize
from train import declareParserArguments


def entire_world_plot(df, args=ArgumentParser):
    # Make direction for result plot and dataframe
    Path(f"./visualization_results/entire_world_plots/").mkdir(
        parents=True, exist_ok=True
    )
    # Change date column datatype
    df.date = pd.to_datetime(df["date"])
    # Group data by date column
    df = df.groupby("date").sum()
    # Do some feature engineering on the data
    df = feature_engineering(df)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    # Plot new_cases and new_deaths with 7days features
    df[["new_cases", "7_days_MA_new_cases"]].plot(
        ax=axes[0, 0], figsize=(11, 5), alpha=0.5
    )

    df[["new_cases", "7_days_MA_new_cases_per_million"]].plot(
        ax=axes[0, 1], figsize=(11, 5), alpha=0.5
    )

    df[["new_deaths", "7_days_MA_new_deaths"]].plot(
        ax=axes[1, 0], figsize=(11, 5), alpha=0.5
    )

    df[["new_deaths", "7_days_MA_new_deaths_per_million"]].plot(
        ax=axes[1, 1], figsize=(11, 5), alpha=0.5
    )

    # Prevent axis labels from overlapping
    plt.tight_layout()

    # Save the resulting plot
    plt.savefig(
        f"./visualization_results/entire_world_plots/death_and_cases.png",
        bbox_inches="tight",
    )


def location_cases_plot(df, args=ArgumentParser):
    # Make direction for result plot
    Path(f"./visualization_results/LocationPlots").mkdir(parents=True, exist_ok=True)
    # Filter data by location
    df = df[df["location"] == args.location]

    # Create new dataframe for plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    new_data = pd.DataFrame(
        {
            "date": df.date,
            "total_cases": df.total_cases_per_million,
            "new_cases": df.new_cases_per_million,
        }
    )

    # Plot total_cases and new_cases by date
    fig.add_trace(
        go.Scatter(x=new_data.date, y=new_data.total_cases, name="Total Cases")
    )
    fig.add_trace(go.Scatter(x=new_data.date, y=new_data.new_cases, name="New Cases"))

    # Determine the titles in the plot
    fig.update_layout(
        title=f"Number of Cases Covid-19 in {args.location}",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
    )

    # Save the resulting plot
    fig.write_html(f"./visualization_results/LocationPlots/cases.html")


def location_death_plot(df, args=ArgumentParser):
    # Filter data by location
    df = df[df["location"] == args.location]

    # Create new dataframe for plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    new_data = pd.DataFrame(
        {
            "date": df.date,
            "total_deaths": df.total_deaths_per_million,
            "new_deaths": df.new_deaths_per_million,
        }
    )

    # Plot total_deaths_per_million and new_deaths_per_million by date
    fig.add_trace(
        go.Scatter(x=new_data.date, y=new_data.total_deaths, name="Total deaths")
    )
    fig.add_trace(go.Scatter(x=new_data.date, y=new_data.new_deaths, name="New deaths"))
    fig.update_layout(
        title=f"Number of Deaths Covid-19 in {args.location}",
        xaxis_title="Date",
        yaxis_title="Number of deaths",
    )

    # Save the resulting plot
    fig.write_html(f"./visualization_results/LocationPlots/deaths.html")


# Now let's examine whether the vaccine works in Covid-19.
def is_vaccinate_helpful(df, args=ArgumentParser):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Design your column for plot
    booster_ratio = df.total_boosters / df.population
    fully_vacc_ratio = df.people_fully_vaccinated / df.population
    new_death_ratio = normalize(df)

    # Create new dataframe for plot
    new_data = pd.DataFrame(
        {
            "date": df.date,
            "booster_ratio": booster_ratio,
            "fully_vacc_ratio": fully_vacc_ratio,
            "new_deaths": new_death_ratio,
        }
    )

    # Plot
    fig.add_trace(
        go.Scatter(x=new_data.date, y=new_data.booster_ratio, name="Booster Ratio")
    )
    fig.add_trace(
        go.Scatter(
            x=new_data.date, y=new_data.fully_vacc_ratio, name="Fully_Vaccine_ratio"
        )
    )
    fig.add_trace(go.Scatter(x=new_data.date, y=new_data.new_deaths, name="New Deaths"))

    # Design your layouts
    fig.update_layout(
        title=f"Boost shot & Fully Vaccienate efficient? Covid-19 in {args.location}",
        xaxis_title="Date",
        yaxis_title="Number of Deaths",
    )

    # Save the resulting plot
    fig.write_html(f"./visualization_results/LocationPlots/is_vaccinate_helpful.html")


def compare_case_death(df, args=ArgumentParser):
    # Change date column datatype
    df.date = pd.to_datetime(df.date)
    # Fİlter data by location
    df = df[df["location"] == args.location]
    # Sort values by date
    df = df.sort_values(by=["date"], ascending=True, ignore_index=True)

    # Create your fig
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)

    # Set second y-axis
    deaths_ax = ax.twinx()

    # Create labels and title
    ax.set_title(f"Deaths and Cases Covid19 in {args.location}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    deaths_ax.set_ylabel("Deaths")

    # Create death and case lines
    case_line = ax.plot(df.date, df["new_cases"], "c")
    death_line = deaths_ax.plot(df.date, df["new_deaths"], "d")

    # Add legends to plot
    all_lines = case_line + death_line
    ax.legend(all_lines, ["Cases", "Deaths"])

    # Fix unit in plot
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ",")))

    # Save the resulting plot
    plt.savefig(
        f"./visualization_results/LocationPlots/case_death_compare.png",
        bbox_inches="tight",
    )


def visualize(df: pd.DataFrame, args: ArgumentParser) -> NoReturn:
    """With this function, data is visualized with different methods.
    Args:
        df (pd.DataFrame): The given dataframe.
    Returns:
        NoReturn: This method does not return anything."""

    # Plot entire world results and save.
    entire_world_plot(df=df, args=args)
    # Plot cases for location
    location_cases_plot(df=df, args=args)
    # Plot deaths for location
    location_death_plot(df=df, args=args)
    # Is vaccinate helpful? Plot.
    is_vaccinate_helpful(df=df, args=args)
    # Compare cases and deaths
    compare_case_death(df=df, args=args)


if __name__ == "__main__":
    # Declare an ArgumentParser object
    parser = ArgumentParser(description="Visualizatons")
    args = declareParserArguments(parser=parser)

    # Get data from data folder
    df = pd.read_csv(f"./data/{args.data}.csv")

    # visualize
    visualize(df=df, args=args)
