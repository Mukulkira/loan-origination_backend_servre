# In myapp/views.py

import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# Load your TensorFlow model
model = tf.keras.models.load_model('model.h5')
Tokenizer=tf.keras.preprocessing.text.Tokenizer
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences

filename = os.path.join(os.path.dirname(__file__), 'all-data.csv')
df = pd.read_csv(filename, 
                 names=["sentiment", "text"],
                 encoding="utf-8", encoding_errors="replace")
df_y = pd.get_dummies(df.sentiment, dtype=int)

def get_train_data():
    X_train = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test, train_target, test_target = train_test_split(
            df[df.sentiment==sentiment],
            df_y[df.sentiment==sentiment],
            train_size=300,
            test_size=300,
            random_state=42)
   
        X_train.append(train)
    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    return X_train

X_train = get_train_data()

# print("X_train : ")
# print(X_train)

max_words = 10_000
max_len = 128

def tokenize_pad_sequences(text, tokenizer=None):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # Text tokenization
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
        tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer

# Tokenizer adapted to X_train
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train.text)

X_train , tokenizer = tokenize_pad_sequences(X_train.text)

# Define your prediction function
def predict(input_data):
    # Preprocess input_data if needed

    # Tokenize and pad the input data using the tokenizer adapted to X_train
    # input_sequences = tokenizer.texts_to_sequences(input_data)
    # padded_sequences = pad_sequences(input_sequences, padding='post', maxlen=max_len)

    # # Make prediction using the loaded model
    # prediction = model.predict(tokenize_pad_sequences(["This 1975 Honda CR250M Elsinore spent an extended period in storage before it was acquired by the seller in 2021, and a subsequent refurbishment included powder-coating the frame, refinishing the fuel tank and number plates, cleaning and plating various hardw…"],tokenizer)[0])
    # prediction=model.predict(padded_sequences)
    # print(input_data)
    trained_data=tokenize_pad_sequences(input_data,tokenizer)[0]
    # print(trained_data)
    prediction = model.predict(trained_data)
    return prediction


# Create API endpoint for prediction
# @csrf_exempt
# def prediction_endpoint(request):
#     if request.method == 'POST':
#         try:
#             # Get input data from request body
#             data = json.loads(request.body.decode('utf-8'))
#             # print(data)  # Debugging statement
            
#             # Extract 'data' field from JSON
#             data_text = data.get('data')
#             if data_text is None or data_text.strip() == "":
#                 return JsonResponse({'error': 'Empty or missing "data" field in request'}, status=400)
            
#             input_data = np.array([data_text])  # assuming data is in the correct format for your model
#             # Get prediction
#             prediction = predict(input_data)
#             return JsonResponse({'prediction': prediction.tolist()})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=400)
#     else:
#         return JsonResponse({'error': 'Only POST requests are supported.'}, status=400)

import requests
from django.core.cache import cache

# Modify the function to fetch data from the Finnhub API
@csrf_exempt
def prediction_endpoint(request):
    if request.method == 'POST':
        try:
            # Get input data from request body
            data = json.loads(request.body.decode('utf-8'))
            company_name = data.get('company_name')

            # Check if prediction result is in cache
            cache_key = f'prediction:{company_name}'
            cached_result = cache.get(cache_key)
            if cached_result:
                return JsonResponse(cached_result)
            
            
            if company_name is None or company_name.strip() == "":
                return JsonResponse({'error': 'Empty or missing "company_name" field in request'}, status=400)
            
            # Calculate start and end dates for the last one day
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            

            # Fetch data from Finnhub API with user-specified company name
            response = requests.get(f'https://finnhub.io/api/v1/company-news?symbol={company_name}&from={start_date}&to={end_date}&token=cn0628hr01qkcvkfq990cn0628hr01qkcvkfq99g')
            if response.status_code != 200:
                return JsonResponse({'error': 'Failed to fetch data from the Finnhub API'}, status=response.status_code)
            
            data = response.json()
            articles = [{'headline': article['headline'], 'summary': article['summary'], 'url': article['url'], 'image':article['image']} for article in data]  # Extract articles
            
            if not articles:
                return JsonResponse({'error': 'No articles found in the response'}, status=400)
            
            # Get prediction
            prediction = predict([article['summary'] for article in articles])
            
            # Filter articles based on prediction result
            positive_articles = [article for i, article in enumerate(articles) if prediction[i][0] > 0.5]
            negative_articles = [article for i, article in enumerate(articles) if prediction[i][2] > 0.5]

            if not positive_articles:
                return JsonResponse({'error': 'No articles found with positive sentiment prediction confidence greater than 0.5'}, status=400)

            if not negative_articles:
                return JsonResponse({'error': 'No articles found with negative sentiment prediction confidence greater than 0.5'}, status=400)

            # Cache the prediction result for one day
            cache.set(cache_key, {'positive_articles': positive_articles, 'negative_articles': negative_articles}, timeout=24 * 60 * 60)  # Cache for one day (24 hours * 60 minutes * 60 seconds)
            
            return JsonResponse({'positive_articles': positive_articles, 'negative_articles': negative_articles})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are supported.'}, status=400)
