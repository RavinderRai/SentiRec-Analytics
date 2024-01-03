#For this file to run on its own, you will need the averaged_embeddings.csv and glove_model.gensim files in the same directory
#run this line to try it: !python Get_Recommendations.py --model glove_model.gensim --reviews averaged_embeddings.csv

import numpy as np
import pandas as pd
import argparse
from ast import literal_eval
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class HeadphoneRecommendations:
    def __init__(self, model_file, reviews_df):
        self.glove_model = KeyedVectors.load(model_file)
        self.reviews_df = reviews_df
        
    def get_document_embedding(self, text, glove_model):
        tokens = text.lower().split()
        word_embeddings = [glove_model[word] for word in tokens if word in glove_model]
        if not word_embeddings:
            return np.zeros(glove_model.vector_size)
        return np.mean(word_embeddings, axis=0)
    
    def get_recommendation(self, user_input):
        user_input_embedding = self.get_document_embedding(user_input, self.glove_model)
        
        cos_sim = self.reviews_df['ProductEmbedding'].apply(
            lambda x: cosine_similarity([user_input_embedding], [x])[0][0]
        )
        cos_sim = cos_sim.rename("CosineSimilarity")

        rec_df = pd.concat([self.reviews_df, cos_sim], axis=1)

        top_recommendations = rec_df.nlargest(5, 'CosineSimilarity')
        
        #dropping the vector columns
        top_recommendations = top_recommendations.drop(columns=['ProductEmbedding'], axis=1)
        
        #resetting the index to look nicer and also only showing the headphone name column
        top_recommendations = top_recommendations.reset_index(drop = True)['Headphone_Name']
        
        return top_recommendations

    
def main():
    parser = argparse.ArgumentParser(description="Headphone Recommendations Script")
    parser.add_argument("--model", required=True, help="Path to the GloVe model file")
    parser.add_argument("--reviews", required=True, help="Path to the reviews DataFrame (CSV file)")
    args = parser.parse_args()

    # Load the reviews DataFrame
    reviews_df = pd.read_csv(args.reviews, converters={'ProductEmbedding': literal_eval})

    # Initialize the recommendations class
    recommendations = HeadphoneRecommendations(model_file=args.model, reviews_df=reviews_df)

    # Get user input from command-line arguments
    user_input = input("Enter your headphone preferences: ")

    # Get recommendations for the user input
    top_recommendations = recommendations.get_recommendation(user_input)
    print("Top 5 Recommendations:")
    print(top_recommendations)

if __name__ == "__main__":
    main()







