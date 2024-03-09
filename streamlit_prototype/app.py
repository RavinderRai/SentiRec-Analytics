import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sqlalchemy import create_engine, inspect
import ast
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")

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

#getting df's for each aspect and leaving out rows with 0 scores
battery_df = headphones_fact_table[['headphoneName', 'batteryLabel', 'batteryScore']].loc[headphones_fact_table['batteryScore'] != 0]
comfort_df = headphones_fact_table[['headphoneName', 'comfortLabel', 'comfortScore']].loc[headphones_fact_table['comfortScore'] != 0]
noisecancellation_df = headphones_fact_table[['headphoneName', 'noisecancellationLabel', 'noisecancellationScore']].loc[headphones_fact_table['noisecancellationScore'] != 0]
soundquality_df = headphones_fact_table[['headphoneName', 'soundqualityLabel', 'soundqualityScore']].loc[headphones_fact_table['soundqualityScore'] != 0]

assert (battery_df['batteryScore'] == 0).sum() == 0, "There are zero values in the 'batteryScore' column."
assert (comfort_df['comfortScore'] == 0).sum() == 0, "There are zero values in the 'batteryScore' column."
assert (noisecancellation_df['noisecancellationScore'] == 0).sum() == 0, "There are zero values in the 'noisecancellationScore' column."
assert (soundquality_df['soundqualityScore'] == 0).sum() == 0, "There are zero values in the 'soundqualityScore' column."

#Airpods 3 doesn't have values in this df
prod_descriptions = prod_descriptions.drop(prod_descriptions[prod_descriptions['headphoneName'] == 'AirPods 3'].index)
prod_descriptions['starsBreakdown'] = prod_descriptions['starsBreakdown'].apply(ast.literal_eval)

def clean_star_label(xstars):
    if xstars[0] != '1':
        return xstars[0] + ' ' + xstars[1:] + 's'
    else:
        return xstars[0] + ' ' + xstars[1:]
    
# Function to generate features text
def generate_features_text(features):
    features_text = ast.literal_eval(features)
    return st.markdown('\n'.join([f'- {feature}' for feature in features_text]), unsafe_allow_html=True)

def avg_sentiment_scores(selected_headphone, selected_sentiment, battery_df, comfort_df, noisecancellation_df, soundquality_df):
    # Create a grouped bar chart using plotly.graph_objects.Figure
    fig = go.Figure()
    
    # Check if a headphone is selected
    if selected_headphone:
        averages = []
        labels = []

        # Iterate through each DataFrame and calculate the average score for the selected sentiment
        for df in [battery_df, comfort_df, noisecancellation_df, soundquality_df]:
            filtered_df = df[df['headphoneName'] == selected_headphone]

            # Check if the filtered DataFrame is not empty
            if not filtered_df.empty:
                aspect_name = filtered_df.columns[1].replace('Label', '')
                avg_score = filtered_df[filtered_df['{}Label'.format(aspect_name)] == selected_sentiment]['{}Score'.format(aspect_name)].mean()
                averages.append(avg_score)
                labels.append(aspect_name)

        fig.add_trace(go.Bar(
            x=labels,
            y=averages,
            name=selected_sentiment,
            marker=dict(color='blue' if selected_sentiment == 'Positive' else '#D87093'),
            hovertemplate='%{x}: %{y}<br>Sentiment: ' + selected_sentiment  # Add sentiment to hover template
        ))

        # Update layout
        fig.update_layout(
            barmode='group',
            title=f'Average {selected_sentiment} Sentiment Scores for {selected_headphone}',
            xaxis=dict(title='Aspect'),
            yaxis=dict(title='Average Score'),
        )
    else:
        # If no headphone is selected, display an empty figure
        fig.update_layout(title='Please select a headphone to display data.')

    return fig

def sentiment_distribution(selected_headphone, selected_sentiment, click_data):
    # Default aspect
    selected_aspect = 'battery'

    if click_data is not None:
        # Extract the clicked bar information
        selected_aspect = click_data['points'][0]['x']

    # Showing a distribution of the sentiments per aspect
    if selected_aspect == 'battery':
        filtered_df = battery_df[battery_df['headphoneName'] == selected_headphone]
    elif selected_aspect == 'comfort':
        filtered_df = comfort_df[comfort_df['headphoneName'] == selected_headphone]
    elif selected_aspect == 'soundquality':
        filtered_df = soundquality_df[soundquality_df['headphoneName'] == selected_headphone]
    elif selected_aspect == 'noisecancellation':
        filtered_df = noisecancellation_df[noisecancellation_df['headphoneName'] == selected_headphone]

    line_color = 'blue' if selected_sentiment == 'Positive' else '#D87093'

    fig = go.Figure()

    # Iterate through each sentiment label and add a kernel density line to the figure
    scatter_data = filtered_df[filtered_df[f'{selected_aspect}Label'] == selected_sentiment]

    kde = gaussian_kde(scatter_data[f'{selected_aspect}Score'])
    x_vals = np.linspace(scatter_data[f'{selected_aspect}Score'].min(), scatter_data[f'{selected_aspect}Score'].max() + 1, 300)
    y_vals = kde(x_vals)

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name=selected_sentiment,
        line=dict(color=line_color, width=2),
        fill='tozeroy',  # Fill area under the curve
        opacity=0.7
    ))

    # Update layout
    fig.update_layout(
        title=f'Sentiment Distribution for {selected_headphone} ({selected_aspect.replace("_", " ").title()})',
        xaxis_title='Sentiment Score',
        yaxis_title='Density',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h')
    )

    return fig


def main():
    selected_headphone = st.selectbox('Select Headphone', prod_descriptions['headphoneName'])
    
    overall_col, display_col, absa_col = st.columns(3)
    
    with overall_col:
    
        selected_row = prod_descriptions[prod_descriptions['headphoneName'] == selected_headphone].iloc[0]
        brand_name = selected_row['brand']
        star_rating = selected_row['stars']
        review_count = selected_row['reviewsCount']
        st.write(f'Brand Name: {brand_name}')
        st.write(f'Overall Rating: {star_rating}')
        st.write(f'Total Number of Reviews: {review_count}')
        
        stars_labels = [clean_star_label(stars) for stars in list(selected_row['starsBreakdown'].keys())]
        values = list(selected_row['starsBreakdown'].values())
        total = sum(values)
        percentages = [(value / total) * 100 for value in values]

        # Create bar chart
        fig = go.Figure(data=[go.Bar(x=stars_labels, y=percentages)])
        fig.update_layout(title=f'Ratings Distribution for {selected_headphone}', yaxis_title='Percentage')
        st.plotly_chart(fig)
        
        st.subheader("Features:")
        generate_features_text(selected_row['features'])
        
    with display_col:
        st.write('Amazon picture and link')
        
    with absa_col:
        # Display grouped bar chart for aspect-based sentiment analysis
        st.subheader('ABSA')
        selected_sentiment = st.radio('Select Sentiment', ['Positive', 'Negative'])
        absa_chart = avg_sentiment_scores(selected_headphone, selected_sentiment, battery_df, comfort_df, noisecancellation_df, soundquality_df)
        #st.plotly_chart(absa_chart)
        
        event = plotly_events(absa_chart, 'click')
        
        #click_data = st.session_state.get('click_data', None)
        st.write(event)
        
        if event is not None:
            clicked_label = event[0]['x']
            st.write(f"Clicked Label: {clicked_label}")
        
        # Display sentiment distribution
        st.subheader('Sentiment Distribution')
        st.write('Sentiment distribution goes here...')
    
if __name__ == "__main__":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://img.freepik.com/free-photo/vivid-blurred-colorful-wallpaper-background_58702-3764.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center; color: black;'>Sentirec Analytics</h1>", unsafe_allow_html=True)
    main()