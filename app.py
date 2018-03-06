import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
app.css.append_css({"external_url": "https://cdnjs.cloudflare.com/ajax/libs/foundation/5.5.3/css/foundation.min.css"})

# Imports
df_trips = pd.read_csv("data/generated-marketplace-trips.csv")
df_planets = pd.read_csv("data/generated-marketplace-planets.csv")

# Add dataframe columns
df_trips["trip_duration"] = df_trips["trip_ended"] - df_trips["trip_started"]
df_trips["trip_requested_dt"] = pd.to_datetime(
    df_trips["trip_requested"], unit="s"
)
df_trips["trip_requested_month"] = df_trips["trip_requested_dt"].dt.month

# UI elements
planet_names = np.append("All", df_planets["name"].values)
planet_options = [{"label": p, "value": p.lower()} for p in planet_names]

user_types = ["Pilot", "Passenger"]
user_options = [{"label": t, "value": t.lower()} for t in user_types]

trips_months = np.sort(df_trips["trip_requested_month"].unique())
month_options = {str(m): str(m) for m in trips_months}
min_month = trips_months.min()
max_month = trips_months.max()

trips_weeks = df_trips["trip_requested_dt"].dt.week
trips_weeks_counts = trips_weeks.value_counts().sort_index()


# Util functions
def id_to_planet(ids):
    return df_planets.iloc[ids]["name"].values


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
                        "x": trips_weeks_counts.index,
                        "y": trips_weeks_counts
                    }]
                }
            )
        ], className="column small-12 medium-6"),
        html.Div([
            dcc.RadioItems(
                id="user-type-top",
                options=user_options,
                value="pilot",
                labelStyle={"display": "inline-block"}
            ),
            dcc.Dropdown(
                id="user-planet-top",
                options=planet_options,
                searchable=False,
                value="all"
            ),
            html.Div(
                id="user-table-top"
            ),
            dcc.RadioItems(
                id="user-type-bottom",
                options=user_options,
                value="passenger",
                labelStyle={"display": "inline-block"}
            ),
            dcc.Dropdown(
                id="user-planet-bottom",
                options=planet_options,
                searchable=False,
                value="all"
            ),
            html.Div(
                id="user-table-bottom"
            )
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


def filter_df(df, month="all", planet="all"):
    """Returns a copy of a dataframe filtered by month and planet.
    Changes planet ids to names."""

    dff = df.copy(deep=True)
    dff.loc[:, "planet"] = id_to_planet(dff["planet"])
    dff = dff[dff["trip_requested_month"] == month] if month != "all" else dff
    dff = dff[dff["planet"].str.lower() == planet] if planet != "all" else dff
    return dff


def pad_df(df, columns, size, spacer="-"):
    """Returns a copy of a dataframe padded to a minimum size"""

    blanks = [spacer for i in range(size)]
    dff = pd.DataFrame({i: blanks for i in columns})
    for idx, row in df[:min(size, len(df))].iterrows():
        dff.iloc[idx, :] = row
    return dff


def get_table_users(df, user_type, month="all", planet="all", top=3):
    """Returns an html.Table object containing user statistics,
    filtered by month and planet."""

    user_rating = user_type + "_rating"  # example: "pilot_rating"
    user_label = user_type.capitalize() + " ID"  # example: "Pilot ID"

    filtered_columns = [
        user_type, "planet", "trip_completed", "trip_duration",
        "price", user_rating
    ]

    dff = df.copy(deep=True)
    dff = filter_df(dff, month=month, planet=planet)
    dff = dff[filtered_columns]
    dff_ranked = dff.groupby(user_type).agg({
        "planet": "max",
        "trip_completed": "sum",
        user_rating: "mean",
        "price": "sum",
        "trip_duration": "mean"
    }).sort_values("trip_completed", ascending=False)
    dff_ranked = dff_ranked.reset_index()
    dff_ranked["price"] = dff_ranked["price"].map(
        lambda x: "{:,.2f}".format(x))
    dff_ranked["trip_duration"] = dff_ranked["trip_duration"].map(
        lambda x: "{}:{:02d}".format(*divmod(int(x), 60)))
    dff_ranked[user_rating] = dff_ranked[user_rating].round(2)
    dff_ranked = pad_df(dff_ranked, filtered_columns, top)
    dff_ranked = dff_ranked[filtered_columns]

    table_rows = [
        html.Tr([html.Th(h) for h in [
            "", user_label, "Planet", "Trips Completed",
            "Avg Trip Duration", "Total Revenue", "Avg Rating"
        ]])
    ]
    table_rows.extend([
        html.Tr([
            html.Td(i + 1),
            html.Td(dff_ranked.iloc[i][user_type]),
            html.Td(dff_ranked.iloc[i]["planet"]),
            html.Td(dff_ranked.iloc[i]["trip_completed"]),
            html.Td(dff_ranked.iloc[i]["trip_duration"]),
            html.Td(dff_ranked.iloc[i]["price"]),
            html.Td(dff_ranked.iloc[i][user_rating])
        ]) for i in range(top)
    ])
    table = html.Table(table_rows)

    return table


@app.callback(
    Output(component_id="user-table-top", component_property="children"),
    [
        Input(component_id="interval-slider", component_property="value"),
        Input(component_id="user-type-top", component_property="value"),
        Input(component_id="user-planet-top", component_property="value")
    ]
)
def update_user_top(month, user_type, planet):
    return get_table_users(df_trips, user_type, month, planet)


@app.callback(
    Output(component_id="user-table-bottom", component_property="children"),
    [
        Input(component_id="interval-slider", component_property="value"),
        Input(component_id="user-type-bottom", component_property="value"),
        Input(component_id="user-planet-bottom", component_property="value")
    ]
)
def update_user_bottom(month, user_type, planet):
    return get_table_users(df_trips, user_type, month, planet)


if __name__ == "__main__":
    app.run_server(debug=True)
