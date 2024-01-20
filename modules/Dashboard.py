import pandas as pd
from sqlalchemy import create_engine, inspect
import dash
from dash import dcc, html
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
    
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/lux/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    dcc.Graph(
        id='numeric-chart',
        figure=px.scatter(headphones_fact_table, x='batteryScore', y='comfortScore', title='Numeric Dot Chart')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)