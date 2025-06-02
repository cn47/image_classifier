from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def plot_learning_curve(history: dict, output_file: str | None = None) -> None:
    # Prepare data for loss and accuracy
    loss_data = pd.DataFrame(
        {
            "Epoch": list(range(1, len(history["train_loss"]) + 1)),
            "Train Loss": history["train_loss"],
            "Val Loss": history["val_loss"],
        },
    ).melt(id_vars="Epoch", var_name="Type", value_name="Loss")

    acc_data = pd.DataFrame(
        {
            "Epoch": list(range(1, len(history["train_acc"]) + 1)),
            "Train Accuracy": history["train_acc"],
            "Val Accuracy": history["val_acc"],
        },
    ).melt(id_vars="Epoch", var_name="Type", value_name="Accuracy")

    # Create subplots for loss and accuracy
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Curve", "Accuracy Curve"))

    # Add loss data to the first subplot
    for loss_type in loss_data["Type"].unique():
        loss_subset = loss_data[loss_data["Type"] == loss_type]
        fig.add_trace(
            go.Scatter(
                x=loss_subset["Epoch"],
                y=loss_subset["Loss"],
                mode="lines+markers",
                name=loss_type,
            ),
            row=1,
            col=1,
        )

    # Add accuracy data to the second subplot
    for acc_type in acc_data["Type"].unique():
        acc_subset = acc_data[acc_data["Type"] == acc_type]
        fig.add_trace(
            go.Scatter(
                x=acc_subset["Epoch"],
                y=acc_subset["Accuracy"],
                mode="lines+markers",
                name=acc_type,
            ),
            row=1,
            col=2,
        )

    # Update traces to set colors for Train and Valid data
    for trace in fig.data:
        if "Train" in trace.name:
            trace.line.color = "blue"
            trace.marker.color = "blue"
        elif "Val" in trace.name:
            trace.line.color = "red"
            trace.marker.color = "red"

    # Update layout
    fig.update_layout(
        title_text="Loss and Accuracy Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        width=1400,
        height=800,
    )

    # Update x-axis to show only integer ticks
    fig.update_xaxes(dtick=1)

    fig.show()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_file, width=1400, height=800, scale=2)


def plot_classification_report(classification_report: dict, output_file: str | None = None) -> None:
    _df = pd.DataFrame(classification_report).round(3)
    plot_df = _df.drop(columns=["accuracy", "macro avg", "weighted avg"]).T
    plot_df = plot_df.reset_index().rename(columns={"index": "class"})

    # Meltしてデータを整形
    plot_df_melted = plot_df.melt(
        id_vars=["class", "support"],
        value_vars=["precision", "recall"],
        var_name="metric",
        value_name="value",
    )

    # サブプロットを作成
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 棒グラフを右軸に追加
    fig.add_trace(
        go.Bar(
            x=plot_df["class"],
            y=plot_df["support"],
            name="Support",
            opacity=0.6,
            text=plot_df["support"],
            textposition="auto",
        ),
        secondary_y=True,
    )

    # 折れ線グラフを左軸に追加
    for metric in plot_df_melted["metric"].unique():
        metric_data = plot_df_melted[plot_df_melted["metric"] == metric]
        fig.add_trace(
            go.Scatter(
                x=metric_data["class"],
                y=metric_data["value"],
                mode="lines+markers",
                name=metric,
            ),
            secondary_y=False,
        )

    # レイアウトを更新
    fig.update_layout(
        title_text="Metrics Visualization",
        xaxis_title="Class",
        yaxis_title="Value (Precision/Recall)",
        yaxis2_title="Support",
        width=800,
        height=600,
        yaxis2=dict(showgrid=False),  # Disable gridlines for the right y-axis
    )

    print(_df[["accuracy", "macro avg", "weighted avg"]])
    fig.show()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_file, width=800, height=600, scale=2)
