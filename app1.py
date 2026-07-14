import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import time

# Page configuration
st.set_page_config(
    page_title="Cattle Health Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CREATE SYNTHETIC DATASET
# ==========================================================
np.random.seed(42)

data = pd.DataFrame({
    "Body Temperature": np.random.normal(38.5, 0.6, 200),
    "Heart Rate": np.random.normal(72, 8, 200),
    "Respiration Rate": np.random.normal(28, 4, 200),
    "Rumination Time": np.random.normal(500, 60, 200),
    "Activity Level": np.random.normal(300, 50, 200),
    "Weight": np.random.normal(450, 40, 200),
    "Age": np.random.randint(2, 12, 200)
})

# Health Status Labeling Logic
labels = []
for i in range(len(data)):
    if data["Body Temperature"][i] > 39.5 or data["Rumination Time"][i] < 430:
        labels.append(2)  # Diseased
    elif data["Activity Level"][i] < 240:
        labels.append(1)  # At Risk
    else:
        labels.append(0)  # Healthy

data["Health Status"] = labels

# ==========================================================
# MODEL TRAINING
# ==========================================================
X = data.drop("Health Status", axis=1)
y = data["Health Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================================
# MAIN HEADER
# ==========================================================
st.title("🐄 Cattle Health Monitoring System")
st.markdown("---")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Input Parameters", "🔬 Prediction Results", "📈 Model Insights"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/96/000000/cow.png", width=150)
    
    with col2:
        st.markdown("### Enter Cattle Physiological Parameters")
        st.info("Fill in the parameters below to get health prediction")
    
    # Create input form
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("*🌡️ Vital Signs*")
            body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 38.5, step=0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 72, step=1)
            resp_rate = st.number_input("Respiration Rate (breaths/min)", 10, 60, 28, step=1)
        
        with col2:
            st.markdown("*🏃 Behavioral Metrics*")
            rumination = st.number_input("Rumination Time (min/day)", 300, 700, 500, step=10)
            activity = st.number_input("Activity Level", 100, 500, 300, step=10)
        
        with col3:
            st.markdown("*📏 Physical Attributes*")
            weight = st.number_input("Weight (kg)", 300, 700, 450, step=5)
            age = st.number_input("Age (years)", 1, 15, 5, step=1)
        
        submitted = st.form_submit_button("🔍 Predict Health Status", use_container_width=True)

with tab2:
    if submitted:
        with st.spinner("Analyzing cattle health data..."):
            time.sleep(2)  # Simulate processing
            
            # Prepare input data
            input_data = np.array([[body_temp, heart_rate, resp_rate,
                                   rumination, activity, weight, age]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            status_map = {0: "Healthy", 1: "At Risk", 2: "Diseased"}
            status_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
            
            # ==========================================================
            # PREDICTION RESULTS
            # ==========================================================
            st.markdown("## 📊 Prediction Results")
            
            # Health Status with colored metric
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Health Status",
                    value=f"{status_colors[prediction]} {status_map[prediction]}",
                    delta=None
                )
            
            with col2:
                # Calculate risk score
                risk_score = (
                    0.35 * min(body_temp / 40, 1.0) +
                    0.20 * max((72 - heart_rate) / 72, 0) +
                    0.15 * max((500 - rumination) / 500, 0) +
                    0.15 * max((300 - activity) / 300, 0) +
                    0.10 * (resp_rate / 40) +
                    0.05 * (age / 10)
                )
                risk_score = min(max(risk_score, 0), 1) * 100
                
                st.metric(
                    label="Health Risk Score",
                    value=f"{risk_score:.1f}%",
                    delta=None
                )
            
            with col3:
                confidence = max(probabilities) * 100
                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence:.1f}%",
                    delta=None
                )
            
            st.markdown("---")
            
            # Prediction Probabilities with progress bars
            st.markdown("### 📈 Prediction Probabilities")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                st.markdown("*Healthy*")
                st.progress(float(probabilities[0]), text=f"{probabilities[0]:.1%}")
            
            with prob_col2:
                st.markdown("*At Risk*")
                st.progress(float(probabilities[1]), text=f"{probabilities[1]:.1%}")
            
            with prob_col3:
                st.markdown("*Diseased*")
                st.progress(float(probabilities[2]), text=f"{probabilities[2]:.1%}")
            
            st.markdown("---")
            
            # Input Parameters Summary
            st.markdown("### 📋 Input Parameters Summary")
            
            param_cols = st.columns(4)
            
            with param_cols[0]:
                st.metric("Body Temperature", f"{body_temp}°C")
                st.metric("Heart Rate", f"{heart_rate} bpm")
            
            with param_cols[1]:
                st.metric("Respiration Rate", f"{resp_rate}/min")
                st.metric("Rumination Time", f"{rumination} min")
            
            with param_cols[2]:
                st.metric("Activity Level", f"{activity}")
                st.metric("Weight", f"{weight} kg")
            
            with param_cols[3]:
                st.metric("Age", f"{age} years")
            
            st.markdown("---")
            
            # ==========================================================
            # SHAP EXPLANATION
            # ==========================================================
            with st.expander("🔎 SHAP Feature Importance (Click to expand)", expanded=True):
                st.markdown("#### Global Feature Importance")
                
                with st.spinner("Calculating SHAP values..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train[:100])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_train[:100], 
                                     feature_names=X.columns, 
                                     show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("SHAP values show the impact of each feature on the model output")
            
            # ==========================================================
            # LIME LOCAL EXPLANATION
            # ==========================================================
            with st.expander("🧠 LIME Local Explanation (Click to expand)", expanded=True):
                st.markdown("#### Local Feature Contributions")
                
                with st.spinner("Generating LIME explanation..."):
                    lime_explainer = LimeTabularExplainer(
                        training_data=X_train,
                        feature_names=X.columns,
                        class_names=["Healthy", "At Risk", "Diseased"],
                        mode="classification"
                    )
                    
                    explanation = lime_explainer.explain_instance(
                        input_scaled[0],
                        model.predict_proba,
                        num_features=5
                    )
                    
                    # Create a dataframe for LIME results
                    lime_results = []
                    for feature, weight in explanation.as_list():
                        lime_results.append({
                            "Feature": feature,
                            "Impact": weight,
                            "Direction": "Positive" if weight > 0 else "Negative"
                        })
                    
                    lime_df = pd.DataFrame(lime_results)
                    
                    # Display LIME results as a dataframe
                    st.dataframe(
                        lime_df.style.applymap(
                            lambda x: 'color: green' if x == 'Positive' else 'color: red',
                            subset=['Direction']
                        ),
                        use_container_width=True
                    )
                    
                    # Create a bar chart of LIME results
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['green' if w > 0 else 'red' for w in lime_df['Impact']]
                    ax.barh(lime_df['Feature'], lime_df['Impact'], color=colors)
                    ax.set_xlabel('Impact on Prediction')
                    ax.set_title('Feature Contributions to Prediction')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            # Success message
            st.balloons()
            st.success("✅ Analysis complete! Scroll up to view all results.")
    
    else:
        # Show placeholder when no prediction made
        st.info("👈 Please enter parameters in the 'Input Parameters' tab and click 'Predict Health Status'")
        
        # Show dataset preview
        st.markdown("### 📊 Sample Dataset Preview")
        st.dataframe(
            data.head(10).style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

with tab3:
    st.markdown("## 📈 Model Performance Insights")
    
    # Model accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_acc = model.score(X_train, y_train)
        st.metric("Training Accuracy", f"{train_acc:.2%}")
    
    with col2:
        test_acc = model.score(X_test, y_test)
        st.metric("Testing Accuracy", f"{test_acc:.2%}")
    
    with col3:
        st.metric("Model Type", "Random Forest")
        st.metric("Number of Trees", "100")
    
    st.markdown("---")
    
    # Feature importance from Random Forest
    st.markdown("### 🔑 Feature Importance (Random Forest)")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create columns for better visualization
    imp_col1, imp_col2 = st.columns([1, 1])
    
    with imp_col1:
        # Display as dataframe
        st.dataframe(
            feature_importance.style.format({'Importance': '{:.3f}'})
            .bar(subset=['Importance'], color='#4CAF50'),
            use_container_width=True
        )
    
    with imp_col2:
        # Display as bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#4CAF50')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Dataset statistics
    st.markdown("### 📊 Dataset Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Samples", len(data))
    with stat_col2:
        st.metric("Features", len(X.columns))
    with stat_col3:
        st.metric("Healthy", len(data[data["Health Status"] == 0]))
    with stat_col4:
        st.metric("At Risk/Diseased", len(data[data["Health Status"] > 0]))
    
    # Class distribution
    st.markdown("#### Class Distribution")
    class_dist = data["Health Status"].value_counts().sort_index()
    class_dist.index = ["Healthy", "At Risk", "Diseased"]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#4CAF50', '#FFC107', '#F44336']
    ax.bar(class_dist.index, class_dist.values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Health Status Distribution')
    for i, v in enumerate(class_dist.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🐄 Cattle Health Monitoring System | Powered by Machine Learning</p>
        <p style='font-size: 0.8em;'>© 2024 | All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True
)