from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta
import joblib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
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

class ForecastRequest(BaseModel):
    dish: str
    num_days: int

@app.get("/")
def home():
    return {"message": "🍽️ Dish Forecast API is running with FastAPI!"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    dish = request.dish
    num_days = request.num_days

    if dish not in available_dishes:
        raise HTTPException(status_code=400, detail=f"Dish '{dish}' not found")

    if not isinstance(num_days, int) or num_days <= 0:
        raise HTTPException(status_code=400, detail="Invalid number of days")

    df_dish = food_df[food_df['dish_name'] == dish].copy()
    features = ['day_of_week', 'quantity_prepared', 'occupancy', 'event_flag', 'avg_rating']
    target = 'quantity_consumed'

    X = df_dish[features]
    y = df_dish[target]

    if len(X) < 8:
        raise HTTPException(status_code=400, detail="Not enough historical data for this dish")

    # Train and save model
    X_train = X.iloc[:-7]
    y_train = y.iloc[:-7]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"xgb_model_{dish.replace(' ', '_')}.pkl")

    latest = df_dish.iloc[-1]
    future_dates = pd.date_range(start=food_df['date'].max() + timedelta(days=1), periods=num_days)

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

    return {
        "dish": dish,
        "forecast": forecast
    }
