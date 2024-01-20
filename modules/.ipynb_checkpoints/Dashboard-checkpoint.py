import pandas as pd
from sqlalchemy import create_engine, inspect
import ast
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

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
    
#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css']
app = dash.Dash(__name__)

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


app.layout = html.Div(style={'backgroundColor': custom_colors['background'], 'color': custom_colors['text'], 'width': '800px'}, children=[
   
    html.Div(
        id='testing-text-info',
        children=[
            'Testing Text',
            html.Span(id='output-2', children=''),
        ]
    ),
    
    html.Label('Select Headphone:'),
    dcc.Dropdown(
        id='headphone-dropdown',
        options=[{'label': headphone, 'value': headphone} for headphone in prod_descriptions['headphoneName']],
        value=prod_descriptions['headphoneName'][0]
    ),
    
    dcc.Graph(id='rating-distribution'),
    html.Div(id='description-text')
])

@app.callback(
    [Output('rating-distribution', 'figure'),
     Output('description-text', 'children')],
    [Input('headphone-dropdown', 'value')]
)
def update_chart(selected_headphone):
    selected_row = prod_descriptions[prod_descriptions['headphoneName'] == selected_headphone].iloc[0]
    
    # Create bar chart
    fig = {
        'data': [{
            'x': list(selected_row['starsBreakdown'].keys()), 
            'y': list(selected_row['starsBreakdown'].values()), 
            'type': 'bar', 
            #'marker': {'color': color_scale}
        }],
        
        'layout': {'title': f'Ratings Distribution for {selected_headphone}'}
    }
    
    description_text = selected_row['description']
    
    return fig, description_text

if __name__ == '__main__':
    app.run_server(debug=True)