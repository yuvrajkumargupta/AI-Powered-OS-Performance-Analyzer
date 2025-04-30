import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

print("🤖 AI Performance Analysis Starting...\n")

# Enhanced Data Loading with Validation
try:
    df = pd.read_csv('system_data.csv', parse_dates=['timestamp'])
    print("✅ Data loaded successfully")
    print(f"📊 Data Shape: {df.shape}, Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Feature Engineering
    df['time_index'] = np.arange(len(df))
    df['rolling_avg'] = df['CPU Usage (%)'].rolling(window=5).mean()
    df['hour'] = df['timestamp'].dt.hour
except Exception as e:
    print(f"❌ Data loading failed: {str(e)}")
    exit()

# --------------------------
# 1. Enhanced Linear Regression
# --------------------------
print("\n🔮 Predictive Modeling:")
X = df[['time_index', 'hour', 'rolling_avg']].dropna()
y = df.loc[X.index, 'CPU Usage (%)']

model = LinearRegression()
model.fit(X, y)
train_pred = model.predict(X)
mae = mean_absolute_error(y, train_pred)

print(f"📈 Model trained (MAE: {mae:.2f}%)")

# Future Prediction (next 5 minutes at 10s intervals)
last_time = df['timestamp'].iloc[-1]
future_steps = 30  # 5 minutes at 10s intervals
future_df = pd.DataFrame({
    'time_index': np.arange(len(df), len(df) + future_steps),
    'hour': [last_time.hour] * future_steps,
    'rolling_avg': [df['CPU Usage (%)'].iloc[-10:].mean()] * future_steps
})

future_df['prediction'] = model.predict(future_df[['time_index', 'hour', 'rolling_avg']])
future_df['timestamp'] = [last_time + timedelta(seconds=10*i) for i in range(1, future_steps+1)]

# --------------------------
# 2. Anomaly Detection
# --------------------------
print("\n👁️ Anomaly Detection:")
anomaly_model = IsolationForest(contamination=0.1, random_state=42)
anomalies = anomaly_model.fit_predict(df[['CPU Usage (%)', 'rolling_avg']].dropna())
df['anomaly'] = anomalies == -1

current_status = "⚠️ ANOMALY DETECTED" if df['anomaly'].iloc[-1] else "✅ Normal"
print(f"Current System Status: {current_status}")

# --------------------------
# 3. Enhanced Output
# --------------------------
# Save predictions with confidence intervals
future_df['upper_bound'] = future_df['prediction'] + mae
future_df['lower_bound'] = future_df['prediction'] - mae
future_df.to_csv('cpu_predictions.csv', index=False)

# Generate report
report = {
    "last_updated": str(datetime.now()),
    "current_cpu": float(df['CPU Usage (%)'].iloc[-1]),
    "prediction_mae": float(mae),
    "anomaly_status": bool(df['anomaly'].iloc[-1]),
    "next_peak": {
        "time": str(future_df.loc[future_df['prediction'].idxmax(), 'timestamp']),
        "value": float(future_df['prediction'].max())
    }
}

with open('performance_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# --------------------------
# 4. Visualization (Optional)
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['CPU Usage (%)'], label='Historical')
plt.plot(future_df['timestamp'], future_df['prediction'], 'g--', label='Prediction')
plt.fill_between(future_df['timestamp'], 
                 future_df['lower_bound'], 
                 future_df['upper_bound'], 
                 color='green', alpha=0.1)
plt.scatter(df[df['anomaly']]['timestamp'], 
            df[df['anomaly']]['CPU Usage (%)'], 
            c='red', label='Anomalies')
plt.title("CPU Usage Prediction with Anomaly Detection")
plt.legend()
plt.savefig('prediction_plot.png')
plt.close()

print("\n📂 Output Generated:")
print(f"- cpu_predictions.csv (Next {future_steps} steps)")
print(f"- performance_report.json")
print(f"- prediction_plot.png")
print("\n✅ Analysis Complete!")