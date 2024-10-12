import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Function to convert finish time
def convert_finish_time(finish_time):
    if isinstance(finish_time, str) and ':' in finish_time:
        try:
            minutes, seconds = finish_time.split(':')
            total_seconds = int(minutes) * 60 + float(seconds)
            return total_seconds
        except ValueError:
            return None
    return None

# Load the model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Race Results Analysis", page_icon="🏁", layout="wide")
    st.markdown("""
        <style>
        .reportview-container {
            background: white;
        }
        .dataframe {
            background-color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Race Results Analysis (赛果分析)")

    # Load the dataset
    try:
        df = pd.read_csv('race_results_3.csv')
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return

    # Preprocess the data
    df['Finish Time'] = df['Finish Time'].apply(convert_finish_time)
    df['Win Odds'] = pd.to_numeric(df['Win Odds'], errors='coerce')
    df['Race Distance'] = pd.to_numeric(df['Race Distance'], errors='coerce')
    df['Declared Weight'] = pd.to_numeric(df['Declared Weight'], errors='coerce')
    df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce')
    df = df.dropna()

    # Encode 'Race Class'
    le = LabelEncoder()
    df['Race Class Encoded'] = le.fit_transform(df['Race Class'])

    # Filter options
    st.subheader("Filter Data (筛选数据)")

    horse_name_input = st.text_input("Search Horse Name (搜索马名)")
    selected_class_input = st.text_input("Search Race Class (搜索比赛类别)")
    min_win_odds = st.number_input("Minimum Win Odds (最低胜率)", value=float(df['Win Odds'].min()), min_value=float(df['Win Odds'].min()), max_value=float(df['Win Odds'].max()))
    max_win_odds = st.number_input("Maximum Win Odds (最高胜率)", value=float(df['Win Odds'].max()), min_value=float(df['Win Odds'].min()), max_value=float(df['Win Odds'].max()))

    filtered_df = df.copy()

    if horse_name_input:
        filtered_df = filtered_df[filtered_df['Horse Name'].str.contains(horse_name_input, case=False)]

    if selected_class_input:
        filtered_df = filtered_df[filtered_df['Race Class'].str.contains(selected_class_input, case=False)]

    filtered_df = filtered_df[(filtered_df['Win Odds'] >= min_win_odds) & (filtered_df['Win Odds'] <= max_win_odds)]

    # Display processed data
    st.write("Processed Data", filtered_df)

    # Load the model
    model, le = load_model()

    # Input fields for new data
    st.subheader("Make a Prediction (做出预测)")

    win_odds = st.number_input("Win Odds (胜率)", value=3.4)
    actual_weight = st.number_input("Actual Weight (实际体重)", value=134)
    race_distance = st.number_input("Race Distance (比赛距离)", value=1650)
    declared_weight = st.number_input("Declared Weight (申报体重)", value=1108)
    draw = st.number_input("Draw (抽签号)", value=4)
    race_class = st.selectbox("Race Class (比赛类别)", options=df['Race Class'].unique())

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'Win Odds': [win_odds],
        'Actual Weight': [actual_weight],
        'Race Distance': [race_distance],
        'Declared Weight': [declared_weight],
        'Draw': [draw],
        'Race Class Encoded': [le.transform([race_class])[0]]
    })

    if st.button("Predict Speed (预测速度)"):
        predicted_speed = model.predict(new_data)
        total_time = race_distance / predicted_speed
        st.success(f'Predicted Speed: {predicted_speed[0]:.2f} m/s, Total Time: {total_time[0]:.2f} seconds')

    # Feature importances
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.subheader("Feature Importances (特征重要性)")
    st.write(feature_importance)

if __name__ == "__main__":
    main()