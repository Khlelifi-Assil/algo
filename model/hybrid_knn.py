import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report

class HybridProductRecommendationModel:
    def __init__(self, customers_file, products_file, ratings_file, train_size=0.8):
        # Load customer data
        self.customers_df = pd.read_json(customers_file)

        # Load product data
        self.products_df = pd.read_json(products_file)

        # Load ratings data
        self.ratings_df = pd.read_json(ratings_file)
        self.customers_df.rename(columns={'Id': 'CustomerID'}, inplace=True)
        self.products_df.rename(columns={'Id': 'ProductID'}, inplace=True)

        # Merge customer and rating data
        self.user_ratings = pd.merge(self.ratings_df, self.customers_df, on='CustomerID', how='inner')

        # Merge user_ratings with product data
        self.user_product_ratings = pd.merge(self.user_ratings, self.products_df, on='ProductID', how='inner')

        # Create a user-item matrix
        self.user_item_matrix = self.user_product_ratings.pivot_table(index='CustomerID', columns='ProductID', values='Rate')
        self.user_similarity = cosine_similarity(self.user_item_matrix.fillna(0))

        # Split the data into training and testing sets
        self.train_user_item_matrix, self.test_user_item_matrix = train_test_split(self.user_item_matrix.fillna(0), train_size=train_size)

    def collaborative_filtering_recommendations(self, user_id, n=5):
        user_sim_scores = self.user_similarity[user_id]
        similar_users = user_sim_scores.argsort()[::-1][1:]  # Exclude the user itself
        
        recommended_products = []
        
        for similar_user in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user]
            user_ratings = self.user_item_matrix.iloc[user_id]
            
            unrated_products = user_ratings[user_ratings.isnull()].index
            rated_products = user_ratings.dropna().index
            
            recommended = similar_user_ratings[rated_products].sort_values(ascending=False).head(n)
            recommended_products.extend(recommended.index)
            
            if len(recommended_products) >= n:
                break
        
        return recommended_products[:n]

    def content_based_filtering_recommendations(self, user_id, n=5):
        # Implement content-based filtering logic here
        # You can use user demographics or product features to make content-based recommendations
        # For simplicity, let's assume content-based recommendations are based on product categories
        # Replace this logic with your content-based filtering implementation
        user_demographics = self.customers_df[self.customers_df['CustomerID'] == user_id]
        if not user_demographics.empty:
            user_category = user_demographics.iloc[0]['Category']
            content_based_recommendations = self.products_df[self.products_df['Category'] == user_category].head(n)
            return content_based_recommendations['ProductID'].tolist()
        else:
            return []

    def hybrid_recommendations(self, user_id, n=5):
        # Combine collaborative and content-based recommendations
        collaborative_recs = self.collaborative_filtering_recommendations(user_id, n)
        content_based_recs = self.content_based_filtering_recommendations(user_id, n)
        
        # Merge and deduplicate recommendations
        hybrid_recs = list(set(collaborative_recs + content_based_recs))
        
        return hybrid_recs[:n]

    def evaluate_recommendations(self):
        # Initialize empty lists to store actual and predicted ratings
        actual_ratings = []
        predicted_ratings = []

        # Iterate over test users to evaluate recommendations
        for user_id in self.test_user_item_matrix.index:
            actual = self.test_user_item_matrix.loc[user_id].dropna().index.tolist()  # Actual rated products
            recommended = self.hybrid_recommendations(user_id, n=5)  # Hybrid recommendations
            
            actual_ratings.extend(actual)
            predicted_ratings.extend(recommended)

        # Create a confusion matrix
        confusion = confusion_matrix(actual_ratings, predicted_ratings)

        # Calculate precision, accuracy, recall, and F1-score
        precision = precision_score(actual_ratings, predicted_ratings, average='weighted')
        accuracy = accuracy_score(actual_ratings, predicted_ratings)
        recall = recall_score(actual_ratings, predicted_ratings, average='weighted')
        f1 = f1_score(actual_ratings, predicted_ratings, average='weighted')

        # Print results
        print("Confusion Matrix:")
        print(confusion)
        print("\nPrecision:", precision)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # Generate a classification report
        classification_rep = classification_report(actual_ratings, predicted_ratings)
        print("\nClassification Report:\n", classification_rep)

# Create an instance of the hybrid recommendation model
model = HybridProductRecommendationModel('input/dataset/customers.json', 'input/dataset/products.json', 'input/dataset/ratings.json')

# Evaluate the model
model.evaluate_recommendations()
