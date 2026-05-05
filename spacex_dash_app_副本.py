import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
)


spacex_df = pd.read_csv(DATA_URL)
max_payload = spacex_df["Payload Mass (kg)"].max()
min_payload = spacex_df["Payload Mass (kg)"].min()

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("SpaceX Launch Records Dashboard"),
        html.Div(
            [
                html.Label("Launch site"),
                dcc.Dropdown(
                    id="site-dropdown",
                    options=[
                        {"label": "All Sites", "value": "ALL"},
                        *[
                            {"label": site, "value": site}
                            for site in sorted(spacex_df["Launch Site"].unique())
                        ],
                    ],
                    value="ALL",
                    clearable=False,
                ),
            ],
            style={"maxWidth": "720px", "margin": "0 auto 24px auto"},
        ),
        dcc.Graph(id="success-pie-chart"),
        html.Div(
            [
                html.Label("Payload range (kg)"),
                dcc.RangeSlider(
                    id="payload-slider",
                    min=0,
                    max=10000,
                    step=1000,
                    marks={i: f"{i}" for i in range(0, 10001, 2500)},
                    value=[min_payload, max_payload],
                ),
            ],
            style={"maxWidth": "900px", "margin": "24px auto"},
        ),
        dcc.Graph(id="success-payload-scatter-chart"),
    ],
    style={"fontFamily": "Arial, sans-serif", "padding": "28px"},
)


@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def update_pie(selected_site):
    if selected_site == "ALL":
        success_df = (
            spacex_df[spacex_df["class"] == 1]
            .groupby("Launch Site", as_index=False)
            .size()
            .rename(columns={"size": "Successful launches"})
        )
        return px.pie(
            success_df,
            values="Successful launches",
            names="Launch Site",
            title="Total Successful Launches by Site",
        )

    site_df = spacex_df[spacex_df["Launch Site"] == selected_site]
    outcome_df = (
        site_df.groupby("class", as_index=False)
        .size()
        .rename(columns={"size": "Launches"})
    )
    outcome_df["Outcome"] = outcome_df["class"].map(
        {0: "Did not land", 1: "Landed"}
    )
    return px.pie(
        outcome_df,
        values="Launches",
        names="Outcome",
        title=f"Launch Outcomes for {selected_site}",
        color="Outcome",
        color_discrete_map={"Landed": "#2A9D8F", "Did not land": "#E76F51"},
    )


@app.callback(
    Output("success-payload-scatter-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_scatter(selected_site, payload_range):
    low, high = payload_range
    filtered = spacex_df[
        (spacex_df["Payload Mass (kg)"] >= low)
        & (spacex_df["Payload Mass (kg)"] <= high)
    ]
    if selected_site != "ALL":
        filtered = filtered[filtered["Launch Site"] == selected_site]

    return px.scatter(
        filtered,
        x="Payload Mass (kg)",
        y="class",
        color="Booster Version Category",
        hover_data=["Launch Site", "Payload"],
        title="Payload Mass vs. Launch Outcome",
        labels={"class": "Landing outcome (0=failed, 1=success)"},
    )


if __name__ == "__main__":
    app.run_server(debug=True)
