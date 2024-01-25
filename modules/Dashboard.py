import pandas as pd
from sqlalchemy import create_engine, inspect
import ast
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

# Connection parameters
server = 'RAVI-DESKTOP\SQLEXPRESS01'
database = 'SentiRec_Analytics'
username = 'RAVI-DESKTOP\RaviB'

# Connection parameters
driver = 'ODBC+Driver+17+for+SQL+Server'  # Adjust the driver name if needed

# Create an SQLAlchemy engine
engine = create_engine(f"mssql+pyodbc://{server}/{database}?driver={driver}")

dataframes_dict = {}

try:
    # Create an inspector to inspect the database and get the tables names
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # Load each table into a Pandas DataFrame
    for table_name in table_names:
        df = pd.read_sql_table(table_name, con=engine)
        # Display or process the DataFrame as needed
        dataframes_dict[table_name] = df
    


except pd.errors.DatabaseError as e:
    print("Error reading from the database:", e)

finally:
    # Dispose of the engine
    engine.dispose()


headphones_fact_table = dataframes_dict['headphones_fact_table']
prod_descriptions = dataframes_dict['amazon_product_descriptions']

#remember to get rid of nulls, NA's, etc. - ASSERT it
battery_df = headphones_fact_table[['headphoneName', 'batteryLabel', 'batteryScore']]
comfort_df = headphones_fact_table[['headphoneName', 'comfortLabel', 'comfortScore']]
noisecancellation_df = headphones_fact_table[['headphoneName', 'noisecancellationLabel', 'noisecancellationScore']]
soundquality_df = headphones_fact_table[['headphoneName', 'soundqualityLabel', 'soundqualityScore']]
    
#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

color_scale = px.colors.sequential.Viridis
custom_colors = {
    'background': '#9dd0fa',
    'text': '#333333',
    'plot_background': '#000080',
    'plot_border': '#000080',
}

#Airpods 3 doesn't have values in this df
prod_descriptions = prod_descriptions.drop(prod_descriptions[prod_descriptions['headphoneName'] == 'AirPods 3'].index)
prod_descriptions['starsBreakdown'] = prod_descriptions['starsBreakdown'].apply(ast.literal_eval)


app.layout = dbc.Container([
    dbc.Row([
        
            html.H1('SentiRec Analytics'),
            html.Label('Select Headphone:'),
            dcc.Dropdown(
                    id='headphone-dropdown',
                    options=[{'label': headphone, 'value': headphone} for headphone in prod_descriptions['headphoneName']],
                    value=prod_descriptions['headphoneName'][0]
                ),
        ]
    ),
    dbc.Row([
        dbc.Col(
            [html.Div(style={'backgroundColor': custom_colors['background'], 'color': custom_colors['text'], 'width': '800px'}, children=[

            

            html.Div(id='brand-name'),
            html.Div(id='star-rating', className='star-rating'),
            html.Div(id='review-count'),
            dcc.Graph(id='rating-distribution'),
            html.H3("Features:"),
            html.Div(id='features-text', style={'margin-top': '20px'})
            ])
        ], width=4),
        dbc.Col([
            html.H4('Test'),
        ], width=4),
        
        dbc.Col([
            html.H4('ABSA'),
            dcc.Dropdown(
            id='label-dropdown',
            options=[{'label': label, 'value': label} for label in headphones_fact_table['batteryLabel'].unique()],
            value=headphones_fact_table['batteryLabel'].unique()[0],
            multi=False,
            placeholder='Select a Battery Label'
            ),
        dcc.Graph(id='sentiment-distribution')
            ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4('Test2')
        ], width=4),
        dbc.Col([
            html.H4('Test3')
        ], width=4),
        dbc.Col([
            html.H4('Test1'),
        ], width=4)
    ]),
])


@app.callback(
    [Output('brand-name', 'children'),
     Output('star-rating', 'children'),
     Output('review-count', 'children'),
     Output('rating-distribution', 'figure'),
     Output('features-text', 'children')],
    [Input('headphone-dropdown', 'value')]
)
def update_chart(selected_headphone):
    selected_row = prod_descriptions[prod_descriptions['headphoneName'] == selected_headphone].iloc[0]
    
    review_count = selected_row['reviewsCount']
    brand_name = selected_row['brand']
    star_rating = selected_row['stars']
    
    #cleaning the text to get the number of stars labels
    #helper function - need to add a space between number and star, and make star plural
    def clean_star_label(xstars):
        if xstars[0] != '1':
            return xstars[0] + ' ' + xstars[1:] + 's'
        else:
            return xstars[0] + ' ' + xstars[1:]
            
    stars_labels = [clean_star_label(stars) for stars in list(selected_row['starsBreakdown'].keys())]
    
    # Calculate percentages for each value in starsBreakdown
    values = list(selected_row['starsBreakdown'].values())
    total = sum(values)
    percentages = [(value / total) * 100 for value in values]
    
    # Create bar chart
    fig = {
        'data': [{
            'x': stars_labels, 
            'y': percentages, 
            'type': 'bar', 
            #'marker': {'color': color_scale}
        }],
        
        'layout': {
            'title': f'Ratings Distribution for {selected_headphone}',
            'yaxis': {'tickvals': percentages, 'ticktext': [f'{val:.1f}%' for val in percentages]}
        }
    }
    
    #description_text = selected_row['description']
    
    features_text = ast.literal_eval(selected_row['features'])
    features_text = html.Div([dcc.Markdown(f'- {feature}') for feature in features_text])
    
    return f'Brand Name: {brand_name}', f'Overall Rating: {star_rating}', f'Total Number of Reviews: {review_count}', fig, features_text

@app.callback(
    dash.dependencies.Output('sentiment-distribution', 'figure'),
    [dash.dependencies.Input('headphone-dropdown', 'value'),
     dash.dependencies.Input('label-dropdown', 'value')]
)
def update_graph(selected_headphone, selected_label):
    filtered_df = headphones_fact_table[(headphones_fact_table['headphoneName'] == selected_headphone) & (headphones_fact_table['batteryLabel'] == selected_label)]
    
    fig = px.histogram(filtered_df, x='batteryScore', nbins=10, color='batteryLabel',
                       labels={'batteryScore': 'Sentiment Score'}, opacity=0.7)
    
    fig.update_layout(title=f'Sentiment Distribution for {selected_headphone} ({selected_label} Battery)',
                      xaxis_title='Sentiment Score', yaxis_title='Frequency')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)