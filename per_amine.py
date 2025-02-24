import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def load_per_amine():
    df = pd.read_csv("data/per_amine.csv")
    return df

def plot_performance_per_amine():
    df = pd.read_csv('data/per_amine.csv')
    # Group by name
    grouped = df.groupby("name")
    max_r2 = grouped["r2"].max().sort_values(ascending=False).clip(-1, 1)

    # Create figure with two subplots
    fig = go.Figure()

    # Bar chart for max R² values
    fig.add_trace(
        go.Bar(
            x=max_r2.index,
            y=max_r2,
            name="Max R²",
            marker=dict(color="blue"),
        )
    )

    # Line chart for number of points
    counts = grouped.size().reindex(max_r2.index)

    fig.add_trace(
        go.Scatter(
            x=max_r2.index,
            y=counts,
            name="Number of Points",
            yaxis="y2",
            mode="lines+markers",
            marker=dict(color="red"),
        )
    )

    fig.update_layout(
        title="R² Values and Data Points for Each Amine",
        xaxis={"title": "Amine"},
        yaxis={"title": "Max R²"},
        yaxis2={
            "title": "Number of Points",
            "overlaying": "y",
            "side": "right",
        },
    )
    fig.show()


def plot_predictions_per_amine():
    """For each amine,  make a plot showing the actual and predicted
    values. Include the r2 value in the title."""
    df = load_per_amine()
    # Sort by r2
    grouped = df.groupby("name")
    max_r2 = grouped["r2"].max().sort_values(ascending=False)
    n_amines = len(max_r2)
    grouped = {name: grouped.get_group(name) for name in max_r2.index}


    titles = []
    for name, group in grouped.items():
        r2 = group["r2"].iloc[0]
        titles.append(f"{name} (R²={r2:.2f})")

    # Make all the plots on a single page
    fig = make_subplots(
        rows=n_amines,
        cols=1,
        subplot_titles=titles
    )

    index = 0

    for name, group in grouped.items():
        index += 1
        fig.update_xaxes(row=index, col=1, range=[-0.2, 1])
        fig.update_yaxes(range=[250, 510])
        fig.add_trace(go.Scatter
                        (y=group["T (K)"],
                         x=group["y_pred"],
                         mode='markers',
                         marker=dict(color='blue', symbol='x'),
                         name='Predicted'), row=index, col=1)
        fig.add_trace(go.Scatter
                        (y=group["T (K)"],
                         x=group["y_actual"],
                         mode='markers',
                         marker=dict(color='red'),
                         name='Actual'), row=index, col=1)
        r2 = group["r2"].iloc[0]


    fig.update_layout(height=20000, title_text="Side By Side Subplots")
    fig.update_layout(showlegend=False)
    fig.show()
