import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
app.css.append_css({"external_url": "https://cdnjs.cloudflare.com/ajax/libs/foundation/5.5.3/css/foundation.min.css"})

df_trips = pd.read_csv("data/generated-marketplace-trips.csv")
df_planets = pd.read_csv("data/generated-marketplace-planets.csv")

planet_names = np.append(df_planets["name"].values, "All")
planet_options = [{"label": p, "value": p.lower()} for p in planet_names]

user_types = ["Pilot", "Passenger"]
user_options = [{"label": t, "value": t.lower()} for t in user_types]
df_trips["trip_requested_dt"] = pd.to_datetime(df_trips["trip_requested"], unit="s")
df_trips["trip_requested_month"] = df_trips["trip_requested_dt"].dt.month
trips_months = np.sort(df_trips["trip_requested_month"].unique())
month_options = {str(m): str(m) for m in trips_months}
min_month = trips_months.min()
max_month = trips_months.max()

df_trips_week = df_trips["trip_requested_dt"].dt.week
df_trips_week = df_trips_week.value_counts().sort_index()

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div(
                id="summary-table"
            ),
            dcc.Slider(
                id="interval-slider",
                min=min_month,
                max=max_month,
                value=max_month,
                marks=month_options
            ),
            dcc.Graph(
                id="display-trips-period",
                config={"displayModeBar": False},
                figure={
                    "data": [{
                        "type": "bar",
                        "x": df_trips_week.index,
                        "y": df_trips_week
                    }]
                }
            )
        ], className="column small-12 medium-6"),
        html.Div([
            dcc.RadioItems(
                options=user_options,
                value="pilot",
                labelStyle={"display": "inline-block"}
            ),
            dcc.Dropdown(
                options=planet_options,
                value="all"
            ),
            html.Table([
                html.Tr([
                    html.Th(),
                    html.Th("Trips Given"),
                    html.Th("Revenue Generated"),
                    html.Th("Average Rating"),
                    html.Th("Average Trip Duration")
                ]),
                html.Tr([
                    html.Td(),
                    html.Td(),
                    html.Td(),
                    html.Td(),
                    html.Td()
                ])
            ]),
            dcc.RadioItems(
                options=user_options,
                value="passenger",
                labelStyle={"display": "inline-block"}
            ),
            dcc.Dropdown(
                options=planet_options,
                value="all"
            ),
            html.Table([
                html.Tr([
                    html.Th(),
                    html.Th("Trips Given"),
                    html.Th("Revenue Generated"),
                    html.Th("Average Rating"),
                    html.Th("Average Trip Duration")
                ]),
                html.Tr([
                    html.Td(),
                    html.Td(),
                    html.Td(),
                    html.Td(),
                    html.Td()
                ])
            ])
        ], className="column small-12 medium-6")

    ], className="row"),
    html.Div([
        html.Div([
            # dcc.Graph(id="display-xy-corr")
        ], className="column small-12 medium-6"),
        html.Div([
            html.Div([
                html.Div([
                    # dcc.Graph(
                    #     id="display-x-planet"
                    # )
                ], className="column small-12"),
                html.Div([
                    # dcc.Graph(
                    #     id="display-y-planet"
                    # )
                ], className="column small-12")
            ], className="row")
        ], className="column small-12 medium-6"),
    ], className="row")
])


def get_month_summary(df, month):
    """Returns a numpy array containing [num_trips, num_pilots, num_passengers]
    for a given month."""

    trips, pilots, passengers = None, None, None
    if month in list(df["trip_requested_month"]):
        # compare with list because `0 in series` returns True
        dff = df[df["trip_requested_month"] == month]
        trips = len(dff)
        pilots = len(dff["pilot"].unique())
        passengers = len(dff["passenger"].unique())

    return np.array([trips, pilots, passengers])


def generate_summary_table(df, month):
    """Returns an html.Table object comparing num_trips, num_pilots, num_passengers
    between a given month and the previous month."""

    cm = get_month_summary(df_trips, month)
    pm = get_month_summary(df_trips, month - 1)
    mm = np.round((cm * 1.0 / pm), 2) if pm[0] else ["-" for i in range(3)]
    pm = pm if pm[0] else ["-" for i in range(3)]

    table = html.Table([
        html.Tr([html.Th(h)
                for h in ["", "Current", "Previous", "M/M Change"]]),
        html.Tr([html.Td("Trips"),
                html.Td(cm[0]), html.Td(pm[0]), html.Td(mm[0])]),
        html.Tr([html.Td("Pilots"),
                html.Td(cm[1]), html.Td(pm[1]), html.Td(mm[1])]),
        html.Tr([html.Td("Passengers"),
                html.Td(cm[2]), html.Td(pm[2]), html.Td(mm[2])])
    ])

    return table


@app.callback(
    Output(component_id="summary-table", component_property="children"),
    [Input(component_id="interval-slider", component_property="value")]
)
def update_summary_table(month):
    return generate_summary_table(df_trips, month)


if __name__ == "__main__":
    app.run_server(debug=True)
