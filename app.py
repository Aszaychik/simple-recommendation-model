from flask import Flask, render_template, request, jsonify
import pandas as pd
from surprise import Dataset, Reader, KNNBasic

app = Flask(__name__)

# Load the data
data = pd.read_csv('feedback_data.csv') 

# Create a Reader object
reader = Reader(rating_scale=(1, 10))

# Load the data into a Surprise Dataset
data_surprise = Dataset.load_from_df(data[['customer_id', 'item_id', 'feedback_rating']], reader)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)

# Train the model on the entire dataset
trainset = data_surprise.build_full_trainset()
model.fit(trainset)

@app.route('/')
def documentation():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    customer_id = int(request.json['customer_id'])

    # Get item predictions for the customer
    item_predictions = []
    for item_id in set(data['item_id']):
        predicted_rating = model.predict(customer_id, item_id).est
        item_predictions.append({'item_id': item_id, 'predicted_rating': predicted_rating})

    # Sort predictions by predicted rating
    item_predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)

    # Get top recommended items
    top_recommendations = item_predictions[:5]  # Adjust the number of recommendations as needed

    return jsonify({'customer_id': customer_id, 'recommendations': top_recommendations})

if __name__ == '__main__':
    app.run(debug=True)
