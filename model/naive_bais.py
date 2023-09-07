import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB  # Use Gaussian Naive Bayes
import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBayesRecommendationModel:
    def __init__(self, customers_file, products_file, ratings_file, model_filename):
        self.customers_file = customers_file
        self.products_file = products_file
        self.ratings_file = ratings_file
        self.model_filename = model_filename

    def load_data(self):
        # Load data from JSON files and merge them into a single DataFrame
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

        data_df = pd.merge(pd.merge(ratings_df, customers_df, on='CustomerID'), products_df, on='ProductID')
        self.X = data_df[['CustomerID', 'ProductID', 'Age']]
        self.y = data_df['Rate']

    def train_model(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create and train the Gaussian Naive Bayes model
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)

        # Save the trained model to a file
        with open(self.model_filename, 'wb') as model_file:
            pickle.dump(nb_model, model_file)

    def evaluate_model(self):
        # Load the trained model
        with open(self.model_filename, 'rb') as model_file:
            nb_model = pickle.load(model_file)

        # Make predictions
        y_pred = nb_model.predict(self.X)

        # Generate classification report
        report = classification_report(self.y, y_pred, target_names=['1', '2', '3', '4', '5'])
        print(report)
        with open("naive_bayes_resultats.txt", "w") as f:
            f.write(report)

        # Generate and visualize the confusion matrix
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
    model = NaiveBayesRecommendationModel('input/dataset/customers.json',
                                         'input/dataset/products.json',
                                         'input/dataset/ratings.json',
                                         'modele_nb_rs.pkl')
    model.train_and_evaluate()
