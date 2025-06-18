#importing the required libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#loading my models
baseline_model = joblib.load("baseline_regression_rfr.pkl")
tuned_model = joblib.load("tuned_regression_rfr.pkl")
scaler = joblib.load("scaler.pkl")

#Listing the features and their input ranges from my dataset
feature_ranges = {
    "cement ": (50, 700, "float"),
    "blast_furnace_slag": (0, 500, "float"),
    "fly_ash": (0, 300, "float"),
    "water": (100, 500, "float"),
    "superplasticizer": (0, 50, "float"),
    "coarse_aggregate": (500, 1500, "float"),
    "fine_aggregate": (400, 1400, "float"),
    "age": (1, 500, "int")
}

# Set page config
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

# Sticky Footer CSS
st.markdown("""
    <style>
    html, body, .main {
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .appview-container .main .block-container {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        padding-bottom: 80px;  /* More space for two-line footer */
    }

    .button{
            background-color: #AFAFEF
            }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: black;
        border-top: 1px solid #ddd;
        text-align: center;
        padding: 10px 0;
        font-size: 0.85rem;
        color: #AAA;
        line-height: 1.4rem;
        z-index: 1000;
    }

    footer, .st-emotion-cache-zq5wmm.ezrtsby0 {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)


# UI - Header
st.title("Concrete Compressive Strength Predictor")
st.markdown("Enter the mix properties below and select a model to predict the **Concrete Compressive Strength (MPa)**.")

# Tabs
tab1, tab2, tab3 = st.tabs([""
        "🔢 Input & Prediction",
        "📊 Prediction History",
        "📈 Model Metrics"
        ])

#-----Tab 1: Inputs------

with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.radio("Select Model", ("Baseline Model", "Tuned Model"))
    model = baseline_model if model_choice == "Baseline Model" else tuned_model

    #sliders
    st.markdown("""
        <style>
        .stSlider > div > div {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Tune the inputs")

    st.markdown(
    f"<div style='padding: 0.5rem 1rem; background-color: #AFAFCF; "
    "color: white"
    "border-left: 4px solid #1a73e8; border-radius: 5px; margin-bottom: 1rem;'>"
    f"<strong> Selected Model:</strong> {model_choice}"
    "</div>",
    unsafe_allow_html=True
)


    with st.form("prediction_form"):
        input_values = []

        cols = st.columns(2)

        for index, (feature, (min_val, max_val, dtype)) in enumerate(feature_ranges.items()):
            with cols[index%2]:
                label = feature.replace("_", " ").title() + " (kg/m³)" if feature != "age" else "Age (days)"
                
                if (index//2) != 0:
                    st.markdown("----")  # Add divider only after the first slider

                if dtype == "float":
                    value = st.slider(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float((min_val + max_val) / 2),
                        step=0.1
                    )
                else:  # int
                    value = st.slider(
                        label,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int((min_val + max_val) / 2),
                        step=1
                    )
                input_values.append(value)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.columns([2,1,2])[1]:
            
            submit = st.form_submit_button(
        label="🚀 PREDICT CONCRETE STRENGTH",
        type="primary"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction logic
    if submit:
        input_array = np.array([input_values])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        # Save latest prediction
        st.session_state["latest_prediction"] = prediction
        st.session_state["model_used"] = model_choice

        # Initialize history if it doesn't exist
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []

        # Save current entry
        st.session_state["prediction_history"].append({
            **{k: v for k, v in zip(feature_ranges.keys(), input_values)},
            "Model": model_choice,
            "Predicted Strength (MPa)": round(prediction, 2)
        })

        st.success(f" Predicted Concrete Compressive Strength: **{prediction:.2f} MPa**")

# --- Tab 2: Prediction History ---
with tab2:
    st.subheader("📘 Prediction History")

    if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
        df_history = pd.DataFrame(st.session_state["prediction_history"])

        st.dataframe(df_history, use_container_width=True)

        if st.button("🔁 Reset History"):
            st.session_state["prediction_history"] = []
            st.success("Prediction history cleared.")
    else:
        st.info("No predictions made yet.")

# --- Tab 3: Model Metrics ---
with tab3:
    st.subheader("📊 Model Performance Metrics")

    try:
        ##load data from csv
        df = pd.read_csv("df_scaled.csv")

        # Ensure 'y_true' column exists
        if 'y_true' not in df.columns:
            st.error("'y_true' column not found in test_data.csv.")
        else:
            y_test = df["y_true"].values
            X_test = df.drop(columns=["y_true"]).values

            # Predict with both models
            y_pred_baseline = baseline_model.predict(X_test)
            y_pred_tuned = tuned_model.predict(X_test)

            # Compute metrics for both models
            def compute_metrics(y_true, y_pred):
                return {
                    "r2": r2_score(y_true, y_pred),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
                }

            metrics_baseline = compute_metrics(y_test, y_pred_baseline)
            metrics_tuned = compute_metrics(y_test, y_pred_tuned)

            st.markdown("---")

            # Plot side-by-side
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred_baseline, alpha=0.6, color="dodgerblue", edgecolors='k')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Strength (MPa)")
                ax.set_ylabel("Predicted Strength (MPa)")
                ax.set_title("Baseline Model: Actual vs. Predicted")
                ax.grid(True)
                with st.columns([1,5,1])[1]:
                    st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred_tuned, alpha=0.6, color="green", edgecolors='k')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Strength (MPa)")
                ax.set_ylabel("Predicted Strength (MPa)")
                ax.set_title("Tuned Model: Actual vs. Predicted")
                ax.grid(True)
                with st.columns([1,5,1])[1]:
                    st.pyplot(fig)

            # Show metrics inside a styled box
            st.markdown(
                """
                <style>
                    .metric-box {
                        display: flex;
                        justify-content: space-around;
                        gap: 2rem;
                        padding: 1.2rem;
                        margin-top: 1rem;
                        background-color: #f9f9f9;
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
                    }
                    .metric-card {
                        text-align: center;
                        flex: 1;
                    }
                    .metric-card h4 {
                        margin-bottom: 0.2rem;
                        font-size: 1rem;
                        color: #444;
                    }
                    .metric-card p {
                        font-size: 1.2rem;
                        font-weight: bold;
                        margin: 0;
                        color: #333;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display metrics for both models inside boxes
            st.markdown("<h4>Baseline Model Metrics</h4>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-card">
                        <h4>R² Score</h4>
                        <p>{metrics_baseline['r2']:.3f}</p>
                    </div>
                    <div class="metric-card">
                        <h4>MAE</h4>
                        <p>{metrics_baseline['mae']:.2f} MPa</p>
                    </div>
                    <div class="metric-card">
                        <h4>RMSE</h4>
                        <p>{metrics_baseline['rmse']:.2f} MPa</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<h4>Tuned Model Metrics</h4>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-card">
                        <h4>R² Score</h4>
                        <p>{metrics_tuned['r2']:.3f}</p>
                    </div>
                    <div class="metric-card">
                        <h4>MAE</h4>
                        <p>{metrics_tuned['mae']:.2f} MPa</p>
                    </div>
                    <div class="metric-card">
                        <h4>RMSE</h4>
                        <p>{metrics_tuned['rmse']:.2f} MPa</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    except FileNotFoundError:
        st.error("Couldn't access the test data.")
    except Exception as e:
        st.error(f"An error occurred while loading test data : {e}")

        st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em; margin-top: 2em;'>
            🔍 <em>Note:</em> The metrics above is based on the training data-set.<br> The Tuned Model was 
             selected after an extensive testing on various testing datasets. <br><br>
            For production use, further validation on unseen data is recommended.
        </div>
        """,
        unsafe_allow_html=True
    )

# Sticky Footer
st.markdown("""
    <div class="footer">
        Note: This app is built for educational purposes. <br>
            Designed by Nirjala | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)
