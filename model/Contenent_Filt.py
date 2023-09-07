import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, customers_file, products_file, ratings_file):
        self.customers_df = pd.read_json(customers_file)
        self.products_df = pd.read_json(products_file)
        self.ratings_df = pd.read_json(ratings_file)
        self.similarity_matrix = None
        self.df = None

    def preprocess_data(self):
        # Merge dataframes to create a unified dataset
        self.customers_df.rename(columns={'Id': 'CustomerID'}, inplace=True)
        self.products_df.rename(columns={'Id': 'ProductID'}, inplace=True)
        merged_df = pd.merge(self.ratings_df, self.customers_df, on='CustomerID', how='inner')
        merged_df = pd.merge(merged_df, self.products_df, on='ProductID', how='inner')

        # Select relevant columns
        self.df = merged_df[['CustomerID', 'ProductID', 'rating', 'Genre', 'Director']]

        # One-hot encoding for Genre and Director
        self.df = pd.get_dummies(self.df, columns=['Genre', 'Director'])

        # Calculate the cosine similarity matrix
        features = self.df.iloc[:, 3:]
        self.similarity_matrix = cosine_similarity(features, features)

    def recommend_product(self, CustomerID, num_recommendations=5):
        # Filter data for the given customer
        customer_data = self.df[self.df['CustomerID'] == CustomerID]

        # Calculate the average rating for the customer's ratings
        avg_rating = customer_data['rating'].mean()

        # Find products that the customer has not rated
        unrated_products = self.df[~self.df['ProductID'].isin(customer_data['ProductID'])]

        # Calculate a weighted score for unrated products based on similarity and average rating
        unrated_products['score'] = unrated_products.apply(
            lambda row: row['Rate'] * avg_rating * self.similarity_matrix[row.name, customer_data.index].mean(),
            axis=1
        )

        # Sort products based on the score
        top_products = unrated_products.sort_values(by='score', ascending=False).head(num_recommendations)

        # Return the recommended product IDs
        recommended_ProductIDs = top_products['ProductID'].tolist()
        return recommended_ProductIDs

# Usage
recommender = ContentBasedRecommender('input/dataset/customers.json', 'input/dataset/products.json', 'input/dataset/ratings.json')
recommender.preprocess_data()

CustomerID_to_recommend = 'customer123'
recommended_products = recommender.recommend_product(CustomerID_to_recommend)

print(f"Recommended products for Customer '{CustomerID_to_recommend}':")
for ProductID in recommended_products:
    print(ProductID)
