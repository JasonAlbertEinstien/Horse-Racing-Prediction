import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
# Load the dataset
file_name = 'race_results_3.csv'
df = pd.read_csv(file_name)

# Convert 'Finish Time' from 'mm:ss.ss' to total seconds (same as before)
def convert_finish_time(finish_time):
    if isinstance(finish_time, str) and ':' in finish_time:
        try:
            minutes, seconds = finish_time.split(':')
            total_seconds = int(minutes) * 60 + float(seconds)
            return total_seconds
        except ValueError:
            return None
    return None

df['Finish Time'] = df['Finish Time'].apply(convert_finish_time)

# Convert numeric columns
df['Win Odds'] = pd.to_numeric(df['Win Odds'], errors='coerce')
df['Race Distance'] = pd.to_numeric(df['Race Distance'], errors='coerce')
df['Declared Weight'] = pd.to_numeric(df['Declared Weight'], errors='coerce')
df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')

# Filter out rows with NaN values
required_columns = ['Win Odds', 'Actual Weight', 'Place', 'Finish Time', 'Race Distance', 'Declared Weight', 'Draw', 'Race Class']
df = df.dropna(subset=required_columns)

# Convert 'Actual Weight' and 'Place' to numeric
df['Actual Weight'] = pd.to_numeric(df['Actual Weight'], errors='coerce')
#df['Place'] = pd.to_numeric(df['Place'], errors='coerce')
df = df.dropna()

# Calculate speed (distance per second) and create a new column
df['Speed (m/s)'] = df['Race Distance'] / df['Finish Time']

# Drop rows with NaN values in the new speed column
df = df.dropna(subset=['Speed (m/s)'])

# Encode 'Race Class'
le = LabelEncoder()
df['Race Class Encoded'] = le.fit_transform(df['Race Class'])

# Selecting features and target variable
#X = df[['Win Odds', 'Actual Weight', 'Place', 'Race Distance', 'Declared Weight', 'Draw', 'Race Class Encoded']]
X = df[['Win Odds', 'Actual Weight', 'Race Distance', 'Declared Weight', 'Draw', 'Race Class Encoded']]
y = df['Speed (m/s)']  # Change target to speed

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Example of making predictions with new data
new_data = pd.DataFrame({
    'Win Odds': [3.4],
    'Actual Weight': [134],
    'Race Distance': [1650],
    'Declared Weight': [1108],
    'Draw': [4],
    'Race Class Encoded': [le.transform(['第五班'])[0]]
})

predicted_speed = model.predict(new_data)
total_time = new_data['Race Distance'][0]/predicted_speed
print(f'Predicted Speed: {predicted_speed[0]} m/s , total speed: {total_time}')

# Display feature importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance)

# Save the model and label encoder
joblib.dump(model, 'model.pkl')  # Save the trained model
joblib.dump(le, 'label_encoder.pkl')  # Save the label encoder