import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load external CSS file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Fills in preset values when a scenario button is clicked
def apply_preset(values):
    for k, v in values.items():
        st.session_state[k] = v

# Builds the probability gauge chart
def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 44}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6C63FF"},
            'steps': [
                {'range': [0, 33], 'color': "#FDECEA"},
                {'range': [33, 66], 'color': "#FFF6E0"},
                {'range': [66, 100], 'color': "#E6F9EF"},
            ],
        },
        title={'text': "Purchase Probability", 'font': {'size': 18}}
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20),
                       paper_bgcolor="rgba(0,0,0,0)", font={'color': "#FAFAFA"})
    return fig

st.set_page_config(page_title="ShopSense AI | Purchase Predictor", layout="wide")
load_css("styles.css")

# Load trained model and expected feature columns
model = joblib.load('best_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Header banner
st.markdown("""
    <div class="main-header">
        <h1> ShopSense AI</h1>
        <p>Predict purchase intent in real time before your visitor leaves.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with model performance stats
with st.sidebar:
    st.markdown("Model Performance")
    st.markdown("""
        <div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-value">0.925</div></div>
        <div class="metric-card"><div class="metric-label">Recall (Purchase)</div><div class="metric-value">0.56</div></div>
        <div class="metric-card"><div class="metric-label">F1-Score (Purchase)</div><div class="metric-value">0.64</div></div>
    """, unsafe_allow_html=True)
    st.divider()
    with st.expander("About this tool"):
        st.write("ShopSense AI uses a tuned Random Forest model trained on 12,000+ real "
                  "e-commerce sessions to predict purchase likelihood enabling targeted "
                  "interventions like discount prompts before a visitor leaves.")

tab1, tab2 = st.tabs([" Predict", " Model Insights"])

with tab1:
    st.subheader("Try an Example Session")

    # Preset example scenarios
    presets = {
        "high": dict(page_values=80.0, product_related=45, product_related_duration=900.0,
                     bounce_rates=0.005, exit_rates=0.01, month="Nov",
                     visitor_type="New Visitor", weekend=True),
        "medium": dict(page_values=15.0, product_related=15, product_related_duration=250.0,
                        bounce_rates=0.02, exit_rates=0.03, month="Mar",
                        visitor_type="Returning Visitor", weekend=False),
        "low": dict(page_values=0.0, product_related=2, product_related_duration=20.0,
                     bounce_rates=0.15, exit_rates=0.15, month="May",
                     visitor_type="Returning Visitor", weekend=False),
    }
    ex1, ex2, ex3 = st.columns(3)
    ex1.button(" High-Intent Shopper", on_click=apply_preset, args=(presets["high"],), use_container_width=True)
    ex2.button(" Casual Browser", on_click=apply_preset, args=(presets["medium"],), use_container_width=True)
    ex3.button(" Likely to Bounce", on_click=apply_preset, args=(presets["low"],), use_container_width=True)

    st.subheader("Session Details")
    col1, col2 = st.columns(2)

    # Input widgets for the key features
    with col1:
        page_values = st.slider("Page Values", 0.0, 200.0, 20.0, step=1.0, key="page_values",
                                 help="Higher values indicate pages closely tied to completing a purchase.")
        product_related = st.number_input("Product-Related Pages Viewed", 0, 300, 20, key="product_related")
        product_related_duration = st.number_input("Time on Product Pages (seconds)", 0.0, 3000.0, 300.0, step=10.0, key="product_related_duration")
        bounce_rates = st.slider("Bounce Rate", 0.0, 0.2, 0.02, step=0.005, key="bounce_rates")

    with col2:
        exit_rates = st.slider("Exit Rate", 0.0, 0.2, 0.02, step=0.005, key="exit_rates")
        month = st.selectbox("Month", ["Aug", "Mar", "May", "Nov", "Dec"], key="month")
        visitor_type = st.selectbox("Visitor Type", ["New Visitor", "Returning Visitor"], key="visitor_type")
        weekend = st.checkbox("Visiting on a Weekend?", key="weekend")

    # Input validation
    validation_error = None
    if product_related == 0 and product_related_duration > 0:
        validation_error = "Time on product pages should be 0 if no product-related pages were viewed."
    elif product_related > 0 and product_related_duration == 0:
        validation_error = "If product-related pages were viewed, time spent should be greater than 0."

    if validation_error:
        st.warning(f" {validation_error}")

    predict_clicked = st.button("Predict Purchase Likelihood", type="primary", disabled=bool(validation_error))

    if predict_clicked:
        # Build input row, using defaults for features not shown in the form
        input_data = {
            'Administrative': 0, 'Administrative_Duration': 0.0, 'Informational': 0, 'Informational_Duration': 0.0,
            'ProductRelated': product_related, 'ProductRelated_Duration': product_related_duration,
            'BounceRates': bounce_rates, 'ExitRates': exit_rates, 'PageValues': page_values,
            'OperatingSystems': 2, 'Browser': 2, 'Region': 1, 'TrafficType': 2, 'Weekend': int(weekend),
            f'Month_{month}': 1, 'VisitorType_Returning_Visitor': 1 if visitor_type == "Returning Visitor" else 0,
        }
        # Encode and align columns to match training data
        input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.markdown(f'<div class="result-card purchase-yes"><h2> Likely to Purchase</h2><h3>{probability:.1%} probability</h3></div>', unsafe_allow_html=True)
                st.success(" Recommended action: No intervention needed — visitor shows strong purchase intent.")
            else:
                st.markdown(f'<div class="result-card purchase-no"><h2> Unlikely to Purchase</h2><h3>{probability:.1%} probability</h3></div>', unsafe_allow_html=True)
                st.info(" Recommended action: Consider a discount prompt or live chat to re-engage this visitor.")

        with gauge_col:
            st.plotly_chart(create_gauge(probability), use_container_width=True)

        # Explain the prediction using the top known features
        st.subheader("Why This Prediction?")
        factors = []
        if page_values > 50:
            factors.append(" **High Page Values** — the model's strongest predictor (36.5% importance) shows strong purchase-related engagement.")
        elif page_values < 5:
            factors.append(" **Low Page Values** — the model's strongest predictor (36.5% importance) shows little purchase-related engagement.")
        if exit_rates > 0.05:
            factors.append(" **High Exit Rate** — this visitor is more likely to be browsing away rather than converting.")
        else:
            factors.append(" **Low Exit Rate** — this visitor is staying engaged rather than leaving the site.")
        if product_related_duration > 500:
            factors.append(" **Extended Browsing Time** — longer time on product pages signals stronger purchase interest.")
        elif product_related_duration < 60 and product_related > 0:
            factors.append(" **Short Browsing Time** — limited time spent suggests lower engagement.")

        for f in factors:
            st.markdown(f'<div class="factor-box">{f}</div>', unsafe_allow_html=True)

with tab2:
    # Feature importance chart from the trained model
    st.subheader("What Drives This Model's Predictions?")
    importance_data = pd.DataFrame({
        'Feature': ['PageValues', 'ExitRates', 'ProductRelated_Duration', 'ProductRelated',
                    'Administrative_Duration', 'BounceRates', 'Administrative'],
        'Importance': [0.365, 0.091, 0.090, 0.074, 0.058, 0.057, 0.043]
    }).sort_values('Importance')

    fig = go.Figure(go.Bar(x=importance_data['Importance'], y=importance_data['Feature'],
                            orientation='h', marker_color='#6C63FF'))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={'color': "#FAFAFA"}, xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Overall Purchase Rate in Training Data", "15.6%")
    kpi2.metric("Sessions Analyzed", "12,205")
    st.caption("Based on the Online Shoppers Intention dataset (UCI Machine Learning Repository).")