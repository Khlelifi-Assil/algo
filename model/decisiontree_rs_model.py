import json
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DecisionTreeRecommendationModel:
    def __init__(self, customers_file, products_file, ratings_file, model_filename):
        self.customers_file = customers_file
        self.products_file = products_file
        self.ratings_file = ratings_file
        self.model_filename = model_filename

    def load_data(self):
        with open(self.customers_file, 'r') as f:
            customers_data = json.load(f)

        with open(self.products_file, 'r') as f:
            products_data = json.load(f)

        with open(self.ratings_file, 'r') as f:
            ratings_data = json.load(f)

        customers_df = pd.DataFrame(customers_data)
        products_df = pd.DataFrame(products_data)
        ratings_df = pd.DataFrame(ratings_data)
        customers_df.rename(columns={'Id': 'CustomerID'}, inplace=True)
        products_df.rename(columns={'Id': 'ProductID'}, inplace=True)

        user_item_ratings = pd.merge(pd.merge(ratings_df, customers_df, on='CustomerID'), products_df, on='ProductID')
        label_encoder = LabelEncoder()
        user_item_ratings['Age'] = label_encoder.fit_transform(user_item_ratings['Age'])
        user_item_ratings['Region'] = label_encoder.fit_transform(user_item_ratings['Region'])
        user_item_ratings['Category'] = label_encoder.fit_transform(user_item_ratings['Category'])
        user_item_ratings['genre'] = label_encoder.fit_transform(user_item_ratings['genre'])
        self.X = user_item_ratings[['Age', 'Region', 'Category','genre']]
        self.y = user_item_ratings['Rate']

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        with open(self.model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)

    def evaluate_model(self):
        with open(self.model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        y_pred = model.predict(self.X)  # You can use X_test for evaluation if needed
        report = classification_report(self.y, y_pred, target_names=['1', '2', '3', '4', '5'])
        print(report)
        with open("treedecision_resultats.txt", "w") as f:
            f.write(report)
        conf_matrix = confusion_matrix(self.y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'])
        plt.xlabel('Predictions')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def train_and_evaluate(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()


# Usage
if __name__ == "__main__":
    model = DecisionTreeRecommendationModel('input/dataset/customers.json',
                                           'input/dataset/products.json',
                                           'input/dataset/ratings.json',
                                           'modele_treedecision_rs.pkl')
    model.train_and_evaluate()