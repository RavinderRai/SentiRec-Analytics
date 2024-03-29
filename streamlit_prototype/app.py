import numpy as np
import pandas as pd
import os
import re
from scipy.stats import gaussian_kde
import requests
from ast import literal_eval
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
import streamlit as st

st.set_page_config(page_title="SentiRec Analytics", layout="wide")

#connecting to google client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sentirec-analytics-service-key.json"
client = bigquery.Client()
def get_bq_df(client, table, dataset='reviews_data', project_id='sentirec-analytics'):
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset}.{table}`
    """
    query_job = client.query(query)
    return query_job.to_dataframe()

headphones_fact_table = get_bq_df(client, 'headphones-fact-table')
prod_descriptions = get_bq_df(client, 'amazon_product_descriptions')
yt_df = get_bq_df(client, 'yt_reviews_gen_summaries')


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
prod_descriptions = prod_descriptions.rename({'Headphones': 'headphoneName'}, axis = 1) #forgot to rename headphone column in GCP
prod_descriptions = prod_descriptions.drop(prod_descriptions[prod_descriptions['headphoneName'] == 'AirPods 3'].index)
prod_descriptions['starsBreakdown'] = prod_descriptions['starsBreakdown'].apply(literal_eval)

#reviews_df = dataframes_dict['averaged_embeddings']
reviews_df = get_bq_df(client, 'averaged_embeddings')
reviews_df = reviews_df.drop(reviews_df[reviews_df['headphoneName'] == 'AirPods 3'].index)
reviews_df['ProductEmbedding'] = reviews_df['ProductEmbedding'].apply(literal_eval)


def clean_star_label(xstars):
    if xstars[0] != '1':
        return xstars[0] + ' ' + xstars[1:] + 's'
    else:
        return xstars[0] + ' ' + xstars[1:]
    
# Function to generate features text
def generate_features_text(features):
    features_text = literal_eval(features)
    return '\n\n'.join([f'- {feature}' for feature in features_text])


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

    line_color = '#22F2FF' if selected_sentiment == 'Positive' else '#FF6161'

    fig = go.Figure()

    # Iterate through each sentiment label and add a kernel density line to the figure
    scatter_data = filtered_df[filtered_df[f'{selected_aspect}Label'] == selected_sentiment]
    
    #handles the case when scatter_data is empty - an empty plot will be displayed 
    if len(scatter_data) > 0:
        kde = gaussian_kde(scatter_data[f'{selected_aspect}Score'])
        x_vals = np.linspace(0, 1.2, 300)
        #x_vals = np.linspace(scatter_data[f'{selected_aspect}Score'].min(), scatter_data[f'{selected_aspect}Score'].max() + 1, 300)
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

    proper_name_dct = {'battery': 'Battery',
                       'comfort': 'Comfort',
                       'soundquality': 'Sound Quality',
                       'noisecancellation': 'Noise Cancellation',}
    
    custom_ticks = [0.2, 0.6, 1]
    custom_tick_labels = ['1', '3', '5']

    # Update layout
    fig.update_layout(
        title=dict(
            text="Feature Ratings Distribution<br>               (select above)",
            x=0.16,  # Center horizontally
            font=dict(
                color='rgba(3, 59, 92, 1)'
            )
        ),
        #xaxis_title='Sentiment Score',
        #yaxis_title='Density',
        height=385,
        #width=400,
        plot_bgcolor='rgba(20, 170, 255, 0.1)',
        paper_bgcolor='rgba(0, 0, 255, 0.0)',
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        xaxis=dict(
            tickvals=custom_ticks,
            ticktext=custom_tick_labels,
            tickmode='array',
            tickfont=dict(
                color='rgba(3, 59, 92, 1)',
                size=15
            )
        ),
        yaxis=dict(showgrid=False, showticklabels=False),
    )

    return fig

# Function to get the image filename based on the selected headphone - for local directory
def get_image_filename(headphone_name, image_dir = "translucent_images"):
    return os.path.join(image_dir, f"{headphone_name}.png")

#this one gets image data from image address from internet - using github images for this
def get_image(url):
    response = requests.get(url)
    img_data = response.content
    return img_data

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
        
        .stImage {
            transition: all 0.3s ease-in-out;
        }
        .sidebar-closed .stImage img {
            width: 100%; /* Set the image width to 100% when the sidebar is closed */
        }
        .sidebar-open .stImage img {
            width: 80%; /* Set the image width to 80% when the sidebar is open */
        }
    </style>
"""
def get_yt_summaries(df, row):
    youtuber = df['channel_name'].iloc[row]
    vid_link = df['video_link'].iloc[row]
    summary = df['generated_summaries'].iloc[row]
    cleaned_summary = re.sub(r'\s*\.\s*', '. ', summary)
    return youtuber, vid_link, cleaned_summary


def main(selected_headphone, selected_row):
    #selected_headphone = st.selectbox('Select Headphone', prod_descriptions['headphoneName'])
    
    
    #with st.sidebar.expander("Features"):
    #    generate_features_text(selected_row['features'])
    
    overall_col, display_col, absa_col = st.columns(3)
    
    with overall_col:        
        #padding/white space
        st.write(' ')
        st.write(' ')   
        
        review_count = int(selected_row['reviewsCount'])
        styled_text = f'<div style="font-size: 20px; padding: 10px; border: 0px solid #007bff; border-radius: 5px; text-align: center; color: #033b5c;">Number of Reviews:<br/><span style="font-size: 40px;">{review_count}</span></div>'
        st.write(styled_text, unsafe_allow_html=True)
        
        stars_labels = [clean_star_label(stars) for stars in list(selected_row['starsBreakdown'].keys())]
        values = list(selected_row['starsBreakdown'].values())
        total = sum(values)
        percentages = [(value / total) * 100 for value in values]

        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(x=stars_labels, y=percentages, marker_color='#22F2FF', marker=dict(opacity=0.7))])
        fig.update_layout(
            title=dict(
                text="Ratings Distribution",
                x=0.3,  # Center horizontally
                font=dict(
                    color='rgba(3, 59, 92, 1)'
                )
            ),
            plot_bgcolor='rgba(20, 170, 255, 0.1)',
            paper_bgcolor='rgba(0, 0, 255, 0.0)',
            xaxis=dict(showgrid=False,
                       tickfont=dict(
                            color='rgba(3, 59, 92, 1)',
                            size=15
                        )
            ),
            yaxis=dict(showgrid=False, showticklabels=False),
            #yaxis_title='Percentage',
            height=520,
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
        styled_stars = f'<div style="font-size: 3em; text-align: center; margin-top: -0.5em; color: #033b5c;">{stars}</div>'
        centered_content = f"""
        <div style="text-align: center; margin-bottom: 0.5em;">
            <h3 style="margin-bottom: 0; color: #033b5c;">Overall Rating:</h3>
            {styled_stars}
        </div>
        """
        st.markdown(centered_content, unsafe_allow_html=True)

        brand_name = selected_row['brand']
        centered_content_brand_name = f"""
        <div style="text-align: center; margin-bottom: 0.5em;">
            <h4 style="margin-bottom: 0; font-size: 24px; color: #033b5c;">Brand Name:</h4>
            <span style="font-size: 22px; color: #033b5c;">{brand_name}</span>
        </div>
        """
        st.markdown(centered_content_brand_name, unsafe_allow_html=True)

        
        #displaying image of earbuds
        #image_filename = get_image_filename(selected_headphone)
        
        if selected_headphone == '1MORE Evo':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/1MORE%20Evo.png'
        elif selected_headphone == 'AirPods Pro 2 Earbuds':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/AirPods%20Pro%202%20Earbuds.png'
        elif selected_headphone == 'Beats Fit Pro Earbuds':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Beats%20Fit%20Pro%20Earbuds.png'
        elif selected_headphone == 'Bose Quietcomfort Earbuds':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Bose%20Quietcomfort%20Earbuds.png'
        elif selected_headphone == 'Bose Quietcomfort Earbuds 2':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Bose%20Quietcomfort%20Earbuds%202.png'
        elif selected_headphone == 'Galaxy Buds2 Pro':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Galaxy%20Buds2%20Pro.png'
        elif selected_headphone == 'Jabra Elite 7 Pro':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Jabra%20Elite%207%20Pro.png'
        elif selected_headphone == 'lg tone tf8':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/lg%20tone%20tf8.png'
        elif selected_headphone == 'Pixel Buds Pro':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Pixel%20Buds%20Pro.png'
        elif selected_headphone == 'Sennheiser MTW3':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Sennheiser%20MTW3.png'
        elif selected_headphone == 'Sony Linkbuds original':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Sony%20Linkbuds%20original.png'
        elif selected_headphone == 'Sony Linkbuds S':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Sony%20Linkbuds%20S.png'
        elif selected_headphone == 'Sony WF-1000XM5':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Sony%20WF-1000XM5.png'
        elif selected_headphone == 'sony xm4 earbuds':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/sony%20xm4%20earbuds.png'
        elif selected_headphone == 'Soundcore Liberty 3':
            url = 'https://raw.githubusercontent.com/RavinderRai/SentiRec-Analytics/main/modules/translucent_images/Soundcore%20Liberty%203.png'
        
        columns = st.columns((0.5, 12, 0.5))
        img_data = get_image(url)
        columns[1].image(img_data, caption=None, use_column_width=True)
        #columns[1].image(image_filename, caption=None, use_column_width=True)
            
        st.write('')
        st.write('')
        amazon_link = selected_row['url']
        st.markdown(f"""<h1 style='text-align: center; font-size: 24px;'><a href="{amazon_link}" target="_blank">Amazon Link</a></h1>""", unsafe_allow_html=True)


        
    with absa_col:
        #white space
        st.write('')

        #this centers and fixes everything, including elements in the other columns, to stay centered
        st.markdown("<style>div[data-testid='stHorizontalBlock']>div{display: flex; justify-content: center;}</style>", unsafe_allow_html=True)
        
        styled_text_absa = f'<div style="font-size: 20px; padding: 10px; border: 0px solid #007bff; border-radius: 5px; text-align: center; color: #033b5c;">Feature Scores<br/></div>'
        st.write(styled_text_absa, unsafe_allow_html=True)
        
        check_sentiment = st.checkbox(':blue[Toggle Positive or Negative Sentiment]', value=True)
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
                    
            #nan is not equal to itself so this replaces them with 0
            averages = [0 if x != x else x for x in averages]
                    
            sentiment_ratings_out_of_five = [(sentiment_rating * 5) for sentiment_rating in averages]
            sentiment_ratings_out_of_five = [int(round(sentiment_rating)) for sentiment_rating in sentiment_ratings_out_of_five]
            sentiment_ratings_out_of_five = ['★' * sentiment_rating + '☆' * (5 - sentiment_rating) for sentiment_rating in sentiment_ratings_out_of_five]
            
            aspect = 'battery'

            aspects = ['Battery', 'Comfort', 'Noise Cancellation', 'Sound Quality']
            
            
            num_columns = 2

            # Calculate the number of rows
            num_rows = len(labels)
            num_full_sets = num_rows // num_columns
            num_rows_last_set = num_rows % num_columns

            # Create columns
            columns = st.columns(num_columns)

            # Iterate through rows
            for i in range(num_full_sets):
                for j in range(num_columns):
                    index = i * num_columns + j
                    with columns[j]:
                        #st.write(aspects[index])
                        st.write(f'<div style="font-size: 16px; color: #033b5c;">{aspects[index]}</div>', unsafe_allow_html=True)
                        if st.button(f":blue[{sentiment_ratings_out_of_five[index]}]", key=labels[index]):
                            aspect = labels[index]

            # Deal with the last set of rows (if not a multiple of num_columns)
            if num_rows_last_set != 0:
                for i in range(num_rows_last_set):
                    index = num_full_sets * num_columns + i
                    with columns[i]:
                        #st.write(aspects[index])
                        st.write(f'<div style="font-size: 16px; color: #033b5c;">{aspects[index]}</div>', unsafe_allow_html=True)
                        if st.button(f":blue[{sentiment_ratings_out_of_five[index]}]", key=labels[index]):
                            aspect = labels[index]
        
        # Display sentiment distribution
        absa_distribution = sentiment_distribution(selected_headphone, selected_sentiment, aspect)
        st.markdown(plot_style, unsafe_allow_html=True)
        st.plotly_chart(absa_distribution, use_container_width=True, className="stPlot")
        #st.plotly_chart(absa_distribution)

#https://w.forfun.com/fetch/09/09fd60caa86f9743e9ebdc98c944a335.jpeg
if __name__ == "__main__":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://t4.ftcdn.net/jpg/03/81/60/07/360_F_381600744_sjFtJGWmismEgTtTcPmxscPlM0fcytfh.jpg');
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
        st.write(f"<span style='font-size: 26px;'>SentiRec Analytics:</span>", unsafe_allow_html=True)
        st.write(f"<span style='font-size: 20px;'>Choose a headphone:</span>", unsafe_allow_html=True)


        selected_headphone = st.selectbox('Select from collection', prod_descriptions['headphoneName'], label_visibility='collapsed')

        st.write('')
        st.write(f"<span style='font-size: 20px;'>Highlight Features:</span>", unsafe_allow_html=True)

        selected_row = prod_descriptions[prod_descriptions['headphoneName'] == selected_headphone].iloc[0]
        st.text_area("Highlight Features:", generate_features_text(selected_row['features']), height=500, label_visibility='collapsed')
    
    main(selected_headphone, selected_row)

    filtered_yt_df = yt_df[yt_df['headphoneName'] == selected_headphone]

    filtered_yt_df = pd.DataFrame(filtered_yt_df)


    
    with st.expander(':blue[See what YouTubers are saying...]'):
        num_columns = 3

        # Calculate the number of rows
        num_rows = len(filtered_yt_df)
        num_full_sets = num_rows // num_columns
        num_rows_last_set = num_rows % num_columns

        # Create columns
        columns = st.columns(num_columns)

        # Iterate through rows
        for i in range(num_full_sets):
            for j in range(num_columns):
                row = i * num_columns + j
                youtuber, vid_link, summary = get_yt_summaries(filtered_yt_df, row)
                with columns[j]:
                    st.write(f"<span style='font-size: 20px; color: #0067b0;'>Youtuber: {youtuber}</span>", unsafe_allow_html=True)
                    st.write(f"<span style='font-size: 20px;'><a href='{vid_link}' target='_blank'>Youtuber's Video Review Direct Link</a></span>", unsafe_allow_html=True)
                    st.write(f"<span style='font-size: 20px; color: #0067b0;'>Video Review Summary:</span>", unsafe_allow_html=True)
                    st.text_area('Summary:', summary, height=350, label_visibility='collapsed')

        # Deal with the last set of rows (if not a multiple of num_columns)
        if num_rows_last_set != 0:
            for i in range(num_rows_last_set):
                row = num_full_sets * num_columns + i
                youtuber, vid_link, summary = get_yt_summaries(filtered_yt_df, row)
                with columns[i]:
                    st.write(f"<span style='font-size: 20px; color: #0067b0;'>Youtuber: {youtuber}</span>", unsafe_allow_html=True)
                    st.write(f"<span style='font-size: 20px;'><a href='{vid_link}' target='_blank'>Youtuber's Video Review Direct Link</a></span>", unsafe_allow_html=True)
                    st.write(f"<span style='font-size: 20px; color: #0067b0;'>Video Review Summary:</span>", unsafe_allow_html=True)
                    st.text_area('Summary:', summary, height=350, label_visibility='collapsed')
    