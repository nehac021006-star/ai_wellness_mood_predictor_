# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "ai_wellness_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "rf_mood_model.joblib"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper: create synthetic dataset if not exists ---
def create_sample_dataset(path, n=200, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "user_id": np.random.randint(1, 20, n),
        "date": pd.date_range(start="2024-01-01", periods=n).astype(str),
        "sleep_hours": np.round(np.random.normal(6.5, 1.2, n), 1).clip(0, 14),
        "mood_score": np.random.randint(1, 6, n),  # 1–5 scale
        "stress_level": np.random.randint(1, 6, n),  # 1–5 scale
        "screen_time_hours": np.round(np.random.normal(5, 2, n), 1).clip(0, 24),
        "steps": np.random.randint(500, 15000, n),
        "social_interactions": np.random.randint(0, 15, n),
        "productivity_score": np.random.randint(1, 11, n)  # 1–10 scale
    })
    df.to_csv(path, index=False)
    return df

# --- Load or create data (cached) ---
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        df = create_sample_dataset(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
    return df

# --- Model loader / trainer (cached resource) ---
@st.cache_resource
def get_model(df):
    # If saved model exists, load it; otherwise train and save
    if MODEL_PATH.exists():
        try:
            model, feature_names = joblib.load(MODEL_PATH)
            return model
        except Exception:
            pass
    # Train model
    X = df[["sleep_hours", "stress_level", "screen_time_hours", "steps", "social_interactions", "productivity_score"]]
    y = df["mood_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump((model, list(X.columns)), MODEL_PATH)
    return model

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="AI Wellness & Mood Predictor", layout="wide")
    df = load_data()

    st.title("🌿 AI Wellness & Mood Predictor")
    st.write(
        "Predict your daily mood (1–5) from lifestyle signals: sleep, stress, screen time, steps, social interaction, productivity."
    )

    # --- Layout: Left (input) and Right (EDA & model) ---
    left, right = st.columns([1, 2])

    with left:
        st.header("Enter Today's Data")
        sleep = st.slider("Sleep Hours", 0.0, 14.0, float(df["sleep_hours"].median()), step=0.1)
        stress = st.slider("Stress Level (1-5)", 1, 5, int(df["stress_level"].median()))
        screen = st.slider("Screen Time Hours", 0.0, 24.0, float(df["screen_time_hours"].median()), step=0.1)
        steps = st.number_input("Steps Walked", min_value=0, max_value=100000, value=int(df["steps"].median()))
        social = st.slider("Social Interactions (count)", 0, 50, int(df["social_interactions"].median()))
        productivity = st.slider("Productivity Score (1-10)", 1, 10, int(df["productivity_score"].median()))
        st.write("---")

        if st.button("🔮 Predict Mood"):
            with st.spinner("Loading model..."):
                model = get_model(df)

            features = np.array([[sleep, stress, screen, steps, social, productivity]])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features).max() if hasattr(model, "predict_proba") else 1.0
            st.success(f"Predicted Mood Score: {pred}  —  Confidence: {proba:.2f}")

            st.markdown("**Suggested Actions (based on your inputs):**")
            suggestions = []
            if sleep < 7:
                suggestions.append("Try to increase sleep by 30–60 minutes for better mood.")
            if stress >= 4:
                suggestions.append("Do a short breathing exercise or 10-minute break.")
            if screen > 6:
                suggestions.append("Reduce screen time before sleep (limit blue light).")
            if steps < 5000:
                suggestions.append("Take a short walk (15–20 minutes) to boost mood.")
            if productivity < 5:
                suggestions.append("Break tasks into 25-min focused sprints (Pomodoro).")
            if not suggestions:
                suggestions.append("Keep up the good routine — you're doing well!")
            for s in suggestions:
                st.write("- " + s)

            if pred < 3:
                st.warning("⚠️ Mood is low. Consider small improvements: sleep, short walks, and stress breaks.")
            else:
                st.info("🙂 Mood looks okay. Keep following healthy habits!")

    with right:
        st.header("Dataset Snapshot & EDA")
        st.write("Rows:", df.shape[0], " | Columns:", df.shape[1])
        st.dataframe(df.head(10))

        st.subheader("Distribution: Mood Scores")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="mood_score", data=df, palette="pastel", ax=ax1)
        ax1.set_xlabel("Mood Score (1-5)")
        st.pyplot(fig1)

        st.subheader("Sleep vs Mood")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x="sleep_hours", y="mood_score", data=df, alpha=0.6, ax=ax2)
        ax2.set_xlabel("Sleep Hours")
        ax2.set_ylabel("Mood Score")
        st.pyplot(fig2)

        st.subheader("Stress vs Mood")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="stress_level", y="mood_score", data=df, palette="vlag", ax=ax3)
        ax3.set_xlabel("Stress Level")
        ax3.set_ylabel("Mood Score")
        st.pyplot(fig3)

        st.write("---")
        st.header("Train / Evaluate Model (Quick)")
        if st.button("🔁 Train & Evaluate Model Now"):
            X = df[["sleep_hours", "stress_level", "screen_time_hours", "steps", "social_interactions", "productivity_score"]]
            y = df["mood_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model_local = RandomForestClassifier(n_estimators=200, random_state=42)
            model_local.fit(X_train, y_train)
            ypred = model_local.predict(X_test)
            acc = accuracy_score(y_test, ypred)
            st.write("Accuracy (test):", f"{acc:.3f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, ypred))

            cm = confusion_matrix(y_test, ypred)
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("Actual")
            st.pyplot(fig4)

            importances = pd.Series(model_local.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("Feature Importances")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=importances.values, y=importances.index, palette="mako", ax=ax5)
            st.pyplot(fig5)

            joblib.dump((model_local, list(X.columns)), MODEL_PATH)
            st.success("Model trained and saved.")

        st.subheader("📈 Relationship Between Habits and Mood")
        fig, ax = plt.subplots()
        ax.scatter(df["sleep_hours"], df["mood_score"])
        ax.set_xlabel("Sleep Hours")
        ax.set_ylabel("Mood Score")
        ax.set_title("Sleep vs Mood")
        st.pyplot(fig)

        st.subheader("📂 Upload Your Own Dataset")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            user_df = pd.read_csv(uploaded)
            st.write("Preview of your data:")
            st.dataframe(user_df.head())

    st.write("---")
    st.caption("Made with ❤️  •  AI Wellness & Mood Predictor — by Neha Chauhan")

if __name__ == "__main__":
    main()
