from prophet import Prophet
from database import get_session, Article
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data_for_prophet():
    """Prepare daily sentiment data for Prophet"""
    session = get_session()
    articles = session.query(Article).all()
    session.close()
    
    # Convert to DataFrame
    data = []
    for article in articles:
        if article.sentiment_score is not None and article.published_at:
            data.append({
                'date': article.published_at.date(),
                'sentiment': article.sentiment_score
            })
    
    df = pd.DataFrame(data)
    
    # Group by date and calculate daily average sentiment
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment.columns = ['ds', 'y']
    daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
    
    print(f"Prepared {len(daily_sentiment)} days of data")
    print(f"Date range: {daily_sentiment['ds'].min()} to {daily_sentiment['ds'].max()}")
    
    return daily_sentiment

def build_and_forecast(data, forecast_days=7):
    """Build Prophet model and forecast"""
    
    print(f"\nBuilding Prophet model...")
    
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=False,  # Not enough data for yearly
        weekly_seasonality=False,  # Not enough data for weekly
        daily_seasonality=False,
        interval_width=0.95  # 95% confidence interval
    )
    
    # Fit the model
    model.fit(data)
    print("Model fitted successfully")
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Forecast
    forecast = model.predict(future)
    
    return model, forecast

def analyze_forecast(data, forecast):
    """Analyze and print forecast results"""
    
    print("\n" + "="*60)
    print("SENTIMENT FORECAST (Next 7 Days)")
    print("="*60)
    
    # Get only future forecasts (beyond training data)
    last_date = data['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_date].copy()
    
    for idx, row in future_forecast.iterrows():
        date = row['ds'].strftime('%Y-%m-%d')
        forecast_val = row['yhat']
        lower = row['yhat_lower']
        upper = row['yhat_upper']
        
        # Sentiment interpretation
        if forecast_val > 0.2:
            sentiment = "POSITIVE 🟢"
        elif forecast_val < -0.2:
            sentiment = "NEGATIVE 🔴"
        else:
            sentiment = "NEUTRAL ⚪"
        
        print(f"\n{date}: {sentiment}")
        print(f"  Forecast: {forecast_val:.3f}")
        print(f"  Range: [{lower:.3f}, {upper:.3f}]")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    
    avg_forecast = future_forecast['yhat'].mean()
    current_avg = data['y'].iloc[-7:].mean()
    
    print(f"Current (last 7 days): {current_avg:.3f}")
    print(f"Forecasted (next 7 days): {avg_forecast:.3f}")
    
    if avg_forecast > current_avg:
        print(f"→ Sentiment is expected to IMPROVE by {(avg_forecast - current_avg):.3f}")
    elif avg_forecast < current_avg:
        print(f"→ Sentiment is expected to DECLINE by {(current_avg - avg_forecast):.3f}")
    else:
        print(f"→ Sentiment expected to STABILIZE")

def plot_forecast(model, forecast, data):
    """Plot forecast with confidence intervals"""
    
    fig = model.plot(forecast)
    
    # Add title and labels
    plt.title('AI News Sentiment Forecast (7 Days)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('sentiment_forecast.png', dpi=300, bbox_inches='tight')
    print("\n✓ Forecast plot saved as 'sentiment_forecast.png'")
    
    # Also show components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('forecast_components.png', dpi=300, bbox_inches='tight')
    print("✓ Forecast components saved as 'forecast_components.png'")

def evaluate_model(model, data):
    """Evaluate model performance on historical data"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    # Make predictions on training data
    forecast = model.predict(data[['ds']])
    
    # Calculate metrics
    mae = mean_absolute_error(data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(data['y'], forecast['yhat']))
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("(Lower is better - these measure how far predictions were from actual)")
    print("="*60)

if __name__ == "__main__":
    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    data = prepare_data_for_prophet()
    
    # Step 2: Build model and forecast
    print("\nStep 2: Building model...")
    model, forecast = build_and_forecast(data, forecast_days=7)
    
    # Step 3: Evaluate on historical data
    print("\nStep 3: Evaluating model...")
    evaluate_model(model, data)
    
    # Step 4: Analyze forecast
    print("\nStep 4: Analyzing forecast...")
    analyze_forecast(data, forecast)
    
    # Step 5: Plot
    print("\nStep 5: Generating plots...")
    plot_forecast(model, forecast, data)
    
    print("\n✓ Forecast complete!")