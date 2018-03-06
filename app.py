import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
server = app.server

app.css.append_css({"external_url": "https://fonts.googleapis.com/css?family=Libre+Franklin:500|Space+Mono"})
app.css.append_css({"external_url": "https://cdnjs.cloudflare.com/ajax/libs/foundation/5.5.3/css/foundation.min.css"})
app.css.append_css({"external_url": "http://127.0.0.1:3000/style.css"})

# Imports
df_trips = pd.read_csv("data/generated-marketplace-trips.csv")
df_planets = pd.read_csv("data/generated-marketplace-planets.csv")
df_passengers = pd.read_csv("data/generated-marketplace-passengers.csv")
df_pilots = pd.read_csv("data/generated-marketplace-pilots.csv")

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

df_passengers["created_dt"] = pd.to_datetime(df_passengers["created"], unit="s")
passengers_weeks = df_passengers["created_dt"].dt.week
passengers_weeks_counts = passengers_weeks.value_counts().sort_index()

df_pilots["created_dt"] = pd.to_datetime(df_pilots["created"], unit="s")
pilots_weeks = df_pilots["created_dt"].dt.week
pilots_weeks_counts = pilots_weeks.value_counts().sort_index()


# Util functions
def id_to_planet(ids):
    return df_planets.iloc[ids]["name"].values


app.layout = html.Div([
    html.Div([
        html.Div([
            html.H2(
                id="dash-title",
                className="dash__title"
            )
        ], className="column small-12")
    ], className="row mb--sm"),
    html.Div([
        html.Div([
            html.Div(
                id="summary-table",
                className="summary__table"
            ),
            html.H6(
                "View Summary for Month",
                className="mt--sm mb--xs"
            ),
            dcc.Slider(
                id="interval-slider",
                min=min_month,
                max=max_month,
                value=max_month,
                marks=month_options,
                className="summary__slider mb--md"
            ),
            html.H6(
                "Trip Volume by Week",
                className="mt--sm mb--xs"
            ),
            dcc.Graph(
                id="display-trips-period",
                config={"displayModeBar": False},
                className="summary__histogram mb--sm"
            ),
            html.H6(
                "New Users by Week",
                className="mt--sm mb--xs"
            ),
            dcc.Graph(
                id="display-users-period",
                config={"displayModeBar": False},
                className="summary__histogram mb--sm"
            )
        ], className="column small-12 large-5"),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="user-planet-top",
                        options=planet_options,
                        searchable=False,
                        value="all",
                        className="user__dropdown--top mb--xs"
                    ),
                ], className="column small-5"),
                html.Div([
                    dcc.RadioItems(
                        id="user-type-top",
                        options=user_options,
                        value="pilot",
                        labelStyle={"display": "inline-block"},
                        className="user__radio--top mb--xs"
                    ),
                ], className="column small-7")
            ], className="row"),

            html.Div(
                id="user-table-top",
                className="user__table--top"
            ),

            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="user-planet-bottom",
                        options=planet_options,
                        searchable=False,
                        value="all",
                        className="user__dropdown--bottom mb--xs"
                    ),
                ], className="column small-5"),
                html.Div([
                    dcc.RadioItems(
                        id="user-type-bottom",
                        options=user_options,
                        value="passenger",
                        labelStyle={"display": "inline-block"},
                        className="user__radio--bottom mb--xs"
                    ),
                ], className="column small-7")
            ], className="row"),

            html.Div(
                id="user-table-bottom",
                className="user__table--bottom"
            ),

            dcc.Markdown("""A work in progress.
                Code at [theianchan.github.com](https://theianchan.github.com).
                Data from [Marketplace Data Generation](https://github.com/theianchan/data-notebooks/blob/master/marketplace-data-generation.ipynb).
                """)

        ], className="column small-12 large-7")

    ], className="row"),
    html.Div([
        html.Div([
            # dcc.Graph(id="display-xy-corr")
        ], className="column small-12 large-6"),
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
        ], className="column small-12 large-6"),
    ], className="row")
], className="dash-app__body mt--sm mb--md")


@app.callback(
    Output(component_id="dash-title", component_property="children"),
    [Input(component_id="interval-slider", component_property="value")]
)
def update_dash_title(month):
    return "Rides Overview for Month Starting {}/1".format(month)


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

    cm = ["{:,.0f}".format(x) for x in cm]
    pm = ["{:,.0f}".format(x) if x != "-" else "-" for x in pm]

    table = html.Table([
        html.Tr([html.Th(h)
                for h in ["", "Current Month", "Previous Month", "M/M Change"]]),
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


@app.callback(
    Output(component_id="display-trips-period", component_property="figure"),
    [Input(component_id="interval-slider", component_property="value")]
)
def update_trips_histogram(month):
    data = [
        dict(
            type="bar",
            x=trips_weeks_counts.index,
            y=trips_weeks_counts,
            name="Trips",
            showlegend=False
        )
    ]
    layout = go.Layout(
        autosize=True,
        height=240,
        margin={"r": 20, "b": 30, "l": 40, "t": 20}
    )
    return dict(data=data, layout=layout)


@app.callback(
    Output(component_id="display-users-period", component_property="figure"),
    [Input(component_id="interval-slider", component_property="value")]
)
def update_users_histogram(month):
    data = [
        dict(
            type="scatter",
            mode="lines",
            x=pilots_weeks_counts.index,
            y=pilots_weeks_counts,
            name="Pilots"
        ),
        dict(
            type="scatter",
            mode="lines",
            x=passengers_weeks_counts.index,
            y=passengers_weeks_counts,
            name="Passengers"
        )
    ]
    layout = go.Layout(
        autosize=True,
        height=240,
        margin={"r": 20, "b": 30, "l": 30, "t": 20},
        legend={"orientation": "h"}
    )
    return dict(data=data, layout=layout)


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
    user_label = user_type.capitalize()  # example: "Pilot ID"

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
    dff_ranked.loc["Average " + user_label] = dff_ranked.mean()

    dff_ranked["trip_completed"] = dff_ranked["trip_completed"].round(0)
    dff_ranked["price"] = dff_ranked["price"].map(
        lambda x: "{:,.2f}".format(x))
    dff_ranked["trip_duration"] = dff_ranked["trip_duration"].map(
        lambda x: "{}:{:02d}".format(*divmod(int(x), 60)))
    dff_ranked[user_rating] = dff_ranked[user_rating].round(2)

    dff_ranked_average = dff_ranked.loc["Average " + user_label]
    dff_ranked_average = pd.DataFrame(dff_ranked_average).T
    dff_ranked_average.loc["Average " + user_label, user_type] = "-"
    dff_ranked_average.loc["Average " + user_label, "planet"] = "-"

    dff_ranked = pad_df(dff_ranked, filtered_columns, top)
    dff_ranked = pd.concat((dff_ranked, dff_ranked_average))
    dff_ranked = dff_ranked[filtered_columns]

    table_rows = [
        html.Tr([html.Th(h) for h in [
            "", user_label + " ID", "Planet", "Trips Completed",
            "Avg Trip Duration", "Total Revenue", "Avg Rating"
        ]])
    ]
    table_rows.extend([
        html.Tr([
            html.Td(i + 1 if i < top else "Average " + user_label),
            html.Td(dff_ranked.iloc[i][user_type]),
            html.Td(dff_ranked.iloc[i]["planet"]),
            html.Td(dff_ranked.iloc[i]["trip_completed"]),
            html.Td(dff_ranked.iloc[i]["trip_duration"]),
            html.Td(dff_ranked.iloc[i]["price"]),
            html.Td(dff_ranked.iloc[i][user_rating])
        ]) for i in range(top + 1)
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
