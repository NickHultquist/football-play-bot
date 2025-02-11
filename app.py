import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import streamlit as st

# Step 1: Simulate Data (Replace this with real data)
def create_sample_data():
    data = {
        "coverage": np.random.choice([0, 1, 2, 3], 1000),  # 0: Cover 2, 1: Cover 3, 2: Cover 4, 3: Man
        "formation": np.random.choice([0, 1, 2], 1000),  # 0: Shotgun, 1: I-Form, 2: Singleback
        "routes": np.random.choice([0, 1, 2], 1000),  # 0: 4 Verticals, 1: Slants, 2: Curls
        "yards_gained": np.random.randint(0, 20, 1000)  # Simulated yards gained
    }
    return pd.DataFrame(data)

# Step 2: Train the Model
def train_model(data):
    X = data[["coverage", "formation", "routes"]]
    y = data["yards_gained"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

# Step 3: Generate Plays
def generate_play(model, coverage_type):
    formations = ["Shotgun", "I-Form", "Singleback"]
    routes = ["4 Verticals", "Slants", "Curls"]
    best_play = None
    best_score = -float("inf")

    for formation in formations:
        for route in routes:
            input_data = pd.DataFrame({
                "coverage": [coverage_type],
                "formation": [formations.index(formation)],
                "routes": [routes.index(route)]
            })
            score = model.predict(input_data)[0]
            if score > best_score:
                best_score = score
                best_play = (formation, route)

    return best_play, best_score

# Step 4: Streamlit App
def main():
    st.title("7-on-7 Football Play Generator")

    # Load or create data
    data = create_sample_data()

    # Train the model
    model = train_model(data)

    # Input: Defensive Coverage
    coverage_type = st.selectbox("Select Defensive Coverage", ["Cover 2", "Cover 3", "Cover 4", "Man"])
    coverage_mapping = {"Cover 2": 0, "Cover 3": 1, "Cover 4": 2, "Man": 3}
    coverage_label = coverage_mapping[coverage_type]

    # Generate Play
    if st.button("Generate Play"):
        play, score = generate_play(model, coverage_label)
        st.write(f"**Best Play:** Formation: {play[0]}, Routes: {play[1]}")
        st.write(f"**Predicted Yards Gained:** {score:.2f}")

# Run the app
if __name__ == "__main__":
    main()