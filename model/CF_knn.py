import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report

class RecommendationModel:
    def __init__(self, data_dir='input/dataset'):
        self.data_dir = data_dir
        self.customers_df = pd.read_json(f'{data_dir}/customers.json')
        self.products_df = pd.read_json(f'{data_dir}/products.json')
        self.ratings_df = pd.read_json(f'{data_dir}/ratings.json')
        self.customers_df.rename(columns={'Id': 'CustomerID'}, inplace=True)
        self.products_df.rename(columns={'Id': 'ProductID'}, inplace=True)
        self.user_similarity = None
        self.user_item_matrix = None

    def create_user_item_matrix(self):
        user_ratings = pd.merge(self.ratings_df, self.customers_df, on='CustomerID', how='inner')
        user_product_ratings = pd.merge(user_ratings, self.products_df, on='ProductID', how='inner')
        self.user_item_matrix = user_product_ratings.pivot_table(index='CustomerID', columns='ProductID', values='Rate')

    def train(self, train_size=0.8):
        self.create_user_item_matrix()
        train_user_item_matrix, _ = train_test_split(self.user_item_matrix.fillna(0), train_size=train_size)
        self.user_similarity = cosine_similarity(train_user_item_matrix.fillna(0))

    def recommend_products(self, user_id, n=5):
        if self.user_similarity is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        user_sim_scores = self.user_similarity[user_id]
        similar_users = user_sim_scores.argsort()[::-1][1:]
        
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

    def evaluate(self):
        if self.user_similarity is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        actual_ratings = []
        predicted_ratings = []
        
        for user_id in self.user_item_matrix.index:
            print(user_id)
            actual = self.user_item_matrix.loc[user_id].dropna().index.tolist()
            recommended = self.recommend_products(user_id, n=5)
            
            actual_ratings.extend(actual)
            predicted_ratings.extend(recommended)
        
        confusion = confusion_matrix(actual_ratings, predicted_ratings)
        precision = precision_score(actual_ratings, predicted_ratings, average='weighted')
        accuracy = accuracy_score(actual_ratings, predicted_ratings)
        recall = recall_score(actual_ratings, predicted_ratings, average='weighted')
        f1 = f1_score(actual_ratings, predicted_ratings, average='weighted')
        
        print("Confusion Matrix:")
        print(confusion)
        print("\nPrecision:", precision)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 Score:", f1)
        
        classification_rep = classification_report(actual_ratings, predicted_ratings)
        print("\nClassification Report:\n", classification_rep)

# Example usage:
model = RecommendationModel()
model.train()
model.evaluate()
