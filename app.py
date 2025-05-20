from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import datetime
import joblib

app = Flask(__name__)

# Load and preprocess data once
food_df = pd.read_csv("historical_food_data.csv")
feedback_df = pd.read_csv("user_feedback_data.csv")

food_df['date'] = pd.to_datetime(food_df['date'])
food_df['day_of_week'] = pd.Categorical(food_df['day_of_week']).codes

feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
feedback_df['date'] = feedback_df['timestamp'].dt.date
daily_ratings = feedback_df.groupby('date')['rating'].mean().reset_index()
daily_ratings.columns = ['date', 'avg_rating']
daily_ratings['date'] = pd.to_datetime(daily_ratings['date'])

food_df = food_df.merge(daily_ratings, on='date', how='left')
food_df['avg_rating'].fillna(food_df['avg_rating'].mean(), inplace=True)

available_dishes = food_df['dish_name'].unique()

@app.route('/')
def home():
    return "🍽️ Dish Forecast API is running!"

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    dish = data.get('dish')
    num_days = data.get('num_days')

    if dish not in available_dishes:
        return jsonify({"error": f"Dish '{dish}' not found"}), 400
    if not isinstance(num_days, int) or num_days <= 0:
        return jsonify({"error": "Invalid number of days"}), 400

    df_dish = food_df[food_df['dish_name'] == dish].copy()
    features = ['day_of_week', 'quantity_prepared', 'occupancy', 'event_flag', 'avg_rating']
    target = 'quantity_consumed'

    X = df_dish[features]
    y = df_dish[target]

    X_train = X.iloc[:-7]
    y_train = y.iloc[:-7]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    latest = df_dish.iloc[-1]
    future_dates = pd.date_range(start=food_df['date'].max() + pd.Timedelta(days=1), periods=num_days)

    future_df = pd.DataFrame({
        'day_of_week': future_dates.dayofweek,
        'quantity_prepared': latest['quantity_prepared'],
        'occupancy': latest['occupancy'],
        'event_flag': 0,
        'avg_rating': latest['avg_rating']
    })

    future_preds = model.predict(future_df)

    forecast = [{
    "date": str(date.date()),
    "recommended_quantity_to_prepare": float(round(pred, 2))
} for date, pred in zip(future_dates, future_preds)]


    return jsonify({
        "dish": dish,
        "forecast": forecast
    })

if __name__ == '__main__':
    app.run(debug=True)
