import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate_rnn(customers_file, products_file, ratings_file):
    # Load data from JSON files
    with open(customers_file, 'rb') as f:
        customers_data = json.load(f)
    with open(products_file, 'rb') as f:
        products_data = json.load(f)
    with open(ratings_file, 'rb') as f:
        ratings_data = json.load(f)

    customers_df = pd.DataFrame(customers_data)
    products_df = pd.DataFrame(products_data)
    ratings_df = pd.DataFrame(ratings_data)
    customers_df.rename(columns={'Id': 'CustomerID'}, inplace=True)
    products_df.rename(columns={'Id': 'ProductID'}, inplace=True)
    # Encode customer and product IDs
    customer_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    customers_df['CustomerID'] = customer_encoder.fit_transform(customers_df['CustomerID'])
    products_df['ProductID'] = product_encoder.fit_transform(products_df['ProductID'])
    ratings_df['CustomerID'] = customer_encoder.transform(ratings_df['CustomerID'])
    ratings_df['ProductID'] = product_encoder.transform(ratings_df['ProductID'])

    # Merge dataframes to get the final dataset
    merged_df = pd.merge(ratings_df, customers_df, on='CustomerID')
    merged_df = pd.merge(merged_df, products_df, on='ProductID')

    # Sort data by timestamp (assuming you have a timestamp column)
    merged_df.sort_values(by='CreateDate', inplace=True)

    # Create sequences of user interactions
    sequences = []
    seq_length = 10  # Number of previous interactions to consider

    for _, group in merged_df.groupby('CustomerID'):
        ratings = group['Rate'].values
        if len(ratings) > seq_length:
            for i in range(len(ratings) - seq_length + 1):
                sequences.append(ratings[i:i + seq_length])

    sequences = np.array(sequences)

    # Split data into training and testing sets
    train, test = train_test_split(sequences, test_size=0.2, random_state=42)

    # Create and compile the RNN model
    def create_rnn_model(num_products, seq_length=9):
        input_sequence = Input(shape=(seq_length,), name='input_sequence')
        embedding_layer = Embedding(input_dim=num_products, output_dim=50)(input_sequence)
        lstm_layer = LSTM(50)(embedding_layer)
        output_layer = Dense(1)(lstm_layer)

        model = Model(inputs=input_sequence, outputs=output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    rnn_model = create_rnn_model(num_products=len(product_encoder.classes_))

    # Train the model
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    rnn_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    # Make predictions on the test set
    test_predictions = rnn_model.predict(X_test)

    # Evaluate the model using RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')

    # Round the predictions to the nearest integer to represent ratings
    rounded_predictions = np.round(test_predictions).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, rounded_predictions)
    recall = recall_score(y_test, rounded_predictions, average='weighted')
    f1 = f1_score(y_test, rounded_predictions, average='weighted')
    support = len(y_test)

    # Print metrics
    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Test Recall: {recall:.2f}')
    print(f'Test F1-score: {f1:.2f}')
    print(f'Test Support: {support}')

    # Classification report for test data
    class_report = classification_report(y_test, rounded_predictions)
    print('Test Classification Report:')
    print(class_report)
    with open("rnn_resultats.txt", "w") as f:
            f.write(report)
    rnn_model.save(save_model_path)
    print(f"Model saved to {save_model_path}")

    # Training scores and accuracies
    train_predictions = rnn_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_rounded_predictions = np.round(train_predictions).astype(int)
    train_accuracy = accuracy_score(y_train, train_rounded_predictions)

    # Print training scores and accuracies
    print(f'\nTraining RMSE: {train_rmse:.2f}')
    print(f'Training Accuracy: {train_accuracy:.2f}')

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, rounded_predictions)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.xlabel('Predicted Rating')
    plt.ylabel('True Rating')
    plt.title('Confusion Matrix for Rating Prediction')
    plt.show()
    rnn_model.save(save_model_path)
    print(f"Model saved to {save_model_path}")
# File paths for customers.json, products.json, and ratings.json
customers_file = "input/dataset/customers.json"
products_file = "input/dataset/products.json"
ratings_file = "input/dataset/ratings.json"

save_model_path = "recommendation_rnn_model"

# Call the function to train and evaluate the RNN model
train_and_evaluate_rnn(customers_file, products_file, ratings_file)
