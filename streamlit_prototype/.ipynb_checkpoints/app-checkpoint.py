import numpy as np
import pandas as pd
import seaborn as sns
import os
import requests
from PIL import Image
from io import BytesIO
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
#def generate_features_text(features):
#    features_text = ast.literal_eval(features)
#    st.markdown('\n'.join([f'- {feature}' for feature in features_text]), unsafe_allow_html=True)
    
def generate_features_text(features):
    features_text = ast.literal_eval(features)
    return '\n'.join([f'- {feature}' for feature in features_text])


def avg_sentiment_scores(selected_headphone, selected_sentiment, battery_df, comfort_df, noisecancellation_df, soundquality_df):
    # Create a grouped bar chart using plotly.graph_objects.Figure
    fig = go.FigureWidget()
    
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
            title=f'Average {selected_sentiment} Rating for {selected_headphone}',
            xaxis=dict(title='Aspect'),
            yaxis=dict(title='Average Score'),
            height=400,
            width=400,
            plot_bgcolor='black',
            paper_bgcolor='black'
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
        selected_aspect = click_data
        

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
        height=400,
        width=400,
        legend=dict(x=0, y=1, traceorder='normal', orientation='h')
    )

    return fig

# Function to get the image filename based on the selected headphone
def get_image_filename(headphone_name, image_dir = "translucent_images"):
    return os.path.join(image_dir, f"{headphone_name}.png")

#for plots so they change size when toggling side bar
plot_style = """
    <style>
        .stPlot {
            transition: all 0.3s ease-in-out;
        }
        .sidebar-closed .stPlot {
            width: 100%; /* Set the plot width to 100% when the sidebar is closed */
        }
        .sidebar-open .stPlot {
            width: 80%; /* Set the plot width to 80% when the sidebar is open */
        }
    </style>
"""


def main(selected_headphone, selected_row):
    #selected_headphone = st.selectbox('Select Headphone', prod_descriptions['headphoneName'])
    
    
    #with st.sidebar.expander("Features"):
    #    generate_features_text(selected_row['features'])
    
    overall_col, display_col, absa_col = st.columns(3)
    
    with overall_col:
        #brand_name = selected_row['brand']
        #st.write(f'Brand Name: {brand_name}')
        
        
        review_count = int(selected_row['reviewsCount'])
        styled_text = f'<div style="font-size: 20px; padding: 10px; border: 1px solid #007bff; border-radius: 5px; text-align: center;">Number of Reviews:<br/><span style="font-size: 40px;">{review_count}</span></div>'
        st.write(styled_text, unsafe_allow_html=True)
        
        stars_labels = [clean_star_label(stars) for stars in list(selected_row['starsBreakdown'].keys())]
        values = list(selected_row['starsBreakdown'].values())
        total = sum(values)
        percentages = [(value / total) * 100 for value in values]

        # Create bar chart
        fig = go.Figure(data=[go.Bar(x=stars_labels, y=percentages)])
        fig.update_layout(
            title=f'Ratings Distribution for {selected_headphone}', 
            yaxis_title='Percentage',
            #height=300,
            #width=300
        )

        # Display the CSS style
        st.markdown(plot_style, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, className="stPlot")
        #st.plotly_chart(fig)
        
        
       
        
    with display_col:
        star_rating = selected_row['stars']
        
        # Generate star graphics using Unicode characters
        rating = int(round(star_rating))
        stars = '★' * rating + '☆' * (5 - rating)
        styled_stars = f'<div style="font-size: 3em; text-align: center; margin-top: -0.5em;">{stars}</div>'
        centered_content = f"""
        <div style="text-align: center; margin-bottom: 0.5em;">
            <h3 style="margin-bottom: 0;">Overall Rating:</h3>
            {styled_stars}
        </div>
        """
        st.markdown(centered_content, unsafe_allow_html=True)
        
        image_filename = get_image_filename(selected_headphone)
        
        if os.path.exists(image_filename):
            st.image(image_filename, caption=None)
        else:
            st.write("Image not found.")

        
    with absa_col:
        # Display grouped bar chart for aspect-based sentiment analysis
        #selected_sentiment = st.radio('Select Sentiment', ['Positive', 'Negative'])
        check_sentiment = st.checkbox('Toggle Positive or Negative Sentiment', value=True)
        if check_sentiment:
            selected_sentiment = 'Positive'
        else:
            selected_sentiment = 'Negative'
            
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
                    
            sentiment_ratings_out_of_five = [(sentiment_rating * 5) for sentiment_rating in averages]
            sentiment_ratings_out_of_five = [int(round(sentiment_rating)) for sentiment_rating in sentiment_ratings_out_of_five]
            sentiment_ratings_out_of_five = ['★' * sentiment_rating + '☆' * (5 - sentiment_rating) for sentiment_rating in sentiment_ratings_out_of_five]
            
            col1, col2, col3, col4 = st.columns(4)

            # Define a function to handle the selection
            def select_option(sentiment_ratings_out_of_five):
                # Print the selected option
                st.write(f"You selected: {sentiment_ratings_out_of_five}")

            # Create custom radio buttons using stars
            if col1.button(sentiment_ratings_out_of_five[0], key = labels[0]):
                select_option(sentiment_ratings_out_of_five[0])
            if col2.button(sentiment_ratings_out_of_five[1], key = labels[1]):
                select_option(sentiment_ratings_out_of_five[1])
            if col3.button(sentiment_ratings_out_of_five[2], key = labels[2]):
                select_option(sentiment_ratings_out_of_five[2])
            if col4.button(sentiment_ratings_out_of_five[3], key = labels[3]):
                select_option(sentiment_ratings_out_of_five[3])

            
            
            st.write(labels)
            st.write(sentiment_ratings_out_of_five)
                
            
        absa_chart = avg_sentiment_scores(selected_headphone, selected_sentiment, battery_df, comfort_df, noisecancellation_df, soundquality_df)        
        
        st.markdown(plot_style, unsafe_allow_html=True)
        event = plotly_events(absa_chart, 'click')
        
        if event:
            clicked_label = event[0]['x']
        else:
            clicked_label = 'battery'
        
        # Display sentiment distribution
        st.subheader('Sentiment Distribution')
        absa_distribution = sentiment_distribution(selected_headphone, selected_sentiment, clicked_label)
        st.markdown(plot_style, unsafe_allow_html=True)
        st.plotly_chart(absa_distribution, use_container_width=True, className="stPlot")
        #st.plotly_chart(absa_distribution)
    
import altair as alt
if __name__ == "__main__":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://t4.ftcdn.net/jpg/05/64/29/91/360_F_564299133_Y4DkEzEcleJlHFCjySytI5cEumezC7PB.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    #st.markdown("<h1 style='text-align: center; color: black;'>Sentirec Analytics</h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='margin-top: 0; text-align: center; color: black;'>Sentirec Analytics</h1>", unsafe_allow_html=True)
    #selected_headphone = st.selectbox('Select Headphone', prod_descriptions['headphoneName'])
    with st.sidebar:
        selected_headphone = st.selectbox('Select Headphone', prod_descriptions['headphoneName'])
        selected_row = prod_descriptions[prod_descriptions['headphoneName'] == selected_headphone].iloc[0]
        st.markdown("<h1 style='color: gray;'>Sidebar Title</h1>", unsafe_allow_html=True)
        st.markdown("<h3>This is the sidebar content.</h3>", unsafe_allow_html=True)

        st.text_area("Features:", generate_features_text(selected_row['features']), height=400)
    
    #selected_headphone = st.sidebar.selectbox('SentiRec Analytics: Select Headphone', prod_descriptions['headphoneName'])
    df = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    })

    # Create a plot
    fig = go.Figure(data=go.Scatter(x=df['x'], y=df['y'], mode='markers'))
    


    # Use st.markdown to customize the appearance of the radio buttons
    options = ['★', '★★', '★★★', '★★★★', '★★★★★']
    col1, col2, col3, col4, col5 = st.columns(5)

    # Define a function to handle the selection
    def select_option(option):
        # Print the selected option
        st.write(f"You selected: {option}")

    # Create custom radio buttons using stars
    if col1.button(options[0]):
        select_option(options[0])
    if col2.button(options[1]):
        select_option(options[1])
    if col3.button(options[2]):
        select_option(options[2])
    if col4.button(options[3]):
        select_option(options[3])
    if col5.button(options[4]):
        select_option(options[4])
    
    
    main(selected_headphone, selected_row)
    