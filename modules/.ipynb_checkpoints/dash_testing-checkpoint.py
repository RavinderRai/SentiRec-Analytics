from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


app = Dash(__name__)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4('Zeitraum'),
        ], width=6),
        dbc.Col([
            html.H4('Test'),
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H4('Test2')
        ], width=6),
        dbc.Col([
            html.H4('Test3')
        ], width=6)
    ]),
])

app.run_server(debug=True)