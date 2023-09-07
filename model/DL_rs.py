import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense, LSTM
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Define the RNN model
def create_rnn_model(num_products, seq_length=9):
    input_sequence = Input(shape=(seq_length,), name='input_sequence')
    embedding_layer = Embedding(input_dim=num_products, output_dim=50)(input_sequence)
    lstm_layer = LSTM(50)(embedding_layer)
    output_layer = Dense(1)(lstm_layer)

    model = Model(inputs=input_sequence, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Arrays of train and test files
arrTrainFile=["input/trainset/Rating_0.1_117671_train.json"
         ]

arrTestFile=["input/trainset/Rating_0.1_13083_test.json"
         ]

# Loop through the train and test files
for train_file, test_file in zip(arrTrainFile, arrTestFile):
    # Load data from JSON files
    with open(train_file, 'rb') as f:
        train_data = json.load(f)
    with open(test_file, 'rb') as f:
        test_data = json.load(f)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Encode customer and product IDs
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    train_df['CustomerID'] = user_encoder.fit_transform(train_df['CustomerID'])
    train_df['ProductID'] = product_encoder.fit_transform(train_df['ProductID'])
    test_df['CustomerID'] = user_encoder.transform(test_df['CustomerID'])
    test_df['ProductID'] = product_encoder.transform(test_df['ProductID'])

    # Sort data by timestamp (assuming you have a timestamp column)
    train_df.sort_values(by='CreateDate', inplace=True)
    test_df.sort_values(by='CreateDate', inplace=True)

    # Create sequences of user interactions
    sequences = []
    seq_length = 10  # Number of previous interactions to consider

    for _, group in train_df.groupby('CustomerID'):
        ratings = group['Rate'].values
        if len(ratings) > seq_length:
            for i in range(len(ratings) - seq_length + 1):
                sequences.append(ratings[i:i + seq_length])

    sequences = np.array(sequences)

    # Split data into training and testing sets
    train, test = train_test_split(sequences, test_size=0.2, random_state=42)

    # Create and compile the RNN model
    rnn_model = create_rnn_model(num_products=len(product_encoder.classes_))

    # Train the model
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    rnn_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    # Make predictions on the test set
    test_predictions = rnn_model.predict(X_test)

    # Evaluate the model using RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f'Root Mean Square Error (RMSE) for {train_file} - {test_file}: {rmse:.2f}')


# Evaluate the model using RMSE (Root Mean Square Error)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'Root Mean Square Error (RMSE): {rmse:.2f}')

# Round the predictions to the nearest integer to represent ratings
rounded_predictions = np.round(test_predictions).astype(int)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, rounded_predictions)
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Rating')
plt.ylabel('True Rating')
plt.title('Confusion Matrix for Rating Prediction')
plt.show()