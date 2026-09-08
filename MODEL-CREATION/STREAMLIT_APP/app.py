import streamlit as st
import requests
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Carbon AI Tracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .ai-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "prediction_data" not in st.session_state:

    st.session_state.prediction_data = None


if "ai_response" not in st.session_state:

    st.session_state.ai_response = None


# ============================================================
# API FUNCTIONS
# ============================================================


def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.json()

    except Exception:

        return None


def get_prediction(data):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=30
        )

        if response.status_code == 200:

            return response.json()

        else:

            st.error(
                f"Prediction API Error: "
                f"{response.text}"
            )

            return None

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to AI Service. "
            "Make sure FastAPI is running."
        )

        return None

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

        return None


def ask_ai(data):

    try:

        response = requests.post(
            f"{API_URL}/ask",
            json=data,
            timeout=60
        )

        if response.status_code == 200:

            return response.json()

        else:

            st.error(
                f"AI API Error: "
                f"{response.text}"
            )

            return None

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to AI Service."
        )

        return None

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 Carbon AI")

st.sidebar.markdown(
    "### Supply Chain Carbon Tracker"
)

st.sidebar.markdown("---")


# ------------------------------------------------------------
# API STATUS
# ------------------------------------------------------------

health = check_api()

if health:

    if health.get("xgboost_model"):

        st.sidebar.success(
            "XGBoost: Connected"
        )

    else:

        st.sidebar.error(
            "XGBoost: Not Available"
        )


    if health.get("huggingface"):

        st.sidebar.success(
            "Hugging Face: Connected"
        )

    else:

        st.sidebar.warning(
            "Hugging Face: Not Connected"
        )

else:

    st.sidebar.error(
        "AI Service Offline"
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Carbon Prediction",
        "🤖 Ask AI",
        "📊 Prediction Analysis"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🌱 AI Supply Chain Carbon Tracker'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered carbon emission prediction and analysis'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "AI Model",
            "XGBoost"
        )


    with col2:

        st.metric(
            "Generative AI",
            "Hugging Face"
        )


    with col3:

        st.metric(
            "Backend",
            "FastAPI"
        )


    with col4:

        if health:

            st.metric(
                "Service",
                "Online"
            )

        else:

            st.metric(
                "Service",
                "Offline"
            )


    st.markdown("---")


    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
Supply Chain Data
        ↓
Data Processing
        ↓
XGBoost ML Model
        ↓
Carbon Emission Prediction
        ↓
FastAPI AI Service
        ↓
Hugging Face AI
        ↓
AI Analysis & Recommendations
        ↓
Streamlit Dashboard
        """,
        language="text"
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "System Capabilities"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            """
            ### 🔮 Prediction

            Predict future carbon emissions
            using the trained XGBoost model.
            """
        )


    with col2:

        st.info(
            """
            ### 🤖 AI Analysis

            Ask questions about emission
            predictions using Hugging Face.
            """
        )


    with col3:

        st.info(
            """
            ### 📈 Trend Analysis

            Analyze historical and predicted
            emission trends.
            """
        )


# ============================================================
# CARBON PREDICTION
# ============================================================

elif page == "🔮 Carbon Prediction":

    st.title(
        "🔮 Carbon Emission Prediction"
    )

    st.write(
        "Enter historical emission values "
        "to predict the next emission value."
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2026
        )


        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=9
        )


        time_index = st.number_input(
            "Time Index",
            value=24321
        )


        lag_1 = st.number_input(
            "Previous Emission (Lag 1)",
            value=100.5
        )


    with col2:

        lag_2 = st.number_input(
            "Emission 2 Periods Ago",
            value=98.2
        )


        lag_3 = st.number_input(
            "Emission 3 Periods Ago",
            value=95.7
        )


        rolling_mean_3 = st.number_input(
            "3-Period Rolling Average",
            value=98.13
        )


    st.markdown("")


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Carbon Emission",
        use_container_width=True
    ):

        data = {

            "year": int(year),

            "month": int(month),

            "time_index": int(time_index),

            "lag_1": float(lag_1),

            "lag_2": float(lag_2),

            "lag_3": float(lag_3),

            "rolling_mean_3":
                float(rolling_mean_3)

        }


        with st.spinner(
            "Running XGBoost prediction..."
        ):

            result = get_prediction(
                data
            )


        if result:

            st.session_state.prediction = (
                result
            )

            st.session_state.prediction_data = data


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success(
                "Prediction generated successfully!"
            )


            prediction_value = (
                result
                .get("prediction", {})
                .get("value")
            )


            if prediction_value is not None:

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Predicted CO₂",
                        f"{prediction_value:.4f}"
                    )


                with col2:

                    st.metric(
                        "Previous CO₂",
                        f"{lag_1:.4f}"
                    )


                with col3:

                    change = (
                        prediction_value - lag_1
                    )

                    st.metric(
                        "Change",
                        f"{change:.4f}",
                        delta=f"{change:.4f}"
                    )


            st.markdown("---")


            st.subheader(
                "Prediction Result"
            )

            st.json(result)


# ============================================================
# ASK AI
# ============================================================

elif page == "🤖 Ask AI":

    st.title(
        "🤖 Ask Carbon AI"
    )

    st.write(
        "Ask questions about carbon emissions, "
        "supply chains, or your prediction."
    )


    # --------------------------------------------------------
    # SUBJECT
    # --------------------------------------------------------

    selected_subject = st.selectbox(
        "Select Subject",
        [
            "Carbon Emissions",
            "Supply Chain",
            "Carbon Footprint",
            "Sustainable Supply Chain",
            "Emission Prediction",
            "Carbon Reduction"
        ]
    )


    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_area(
        "Ask your question",
        placeholder=(
            "Example: Is the predicted emission increasing?"
        ),
        height=120
    )


    # --------------------------------------------------------
    # SHOW PREDICTION CONTEXT
    # --------------------------------------------------------

    if st.session_state.prediction:

        prediction_value = (
            st.session_state
            .prediction
            .get("prediction", {})
            .get("value")
        )

        prediction_data = (
            st.session_state
            .prediction_data
        )


        if prediction_value is not None:

            st.info(
                f"Current predicted emission: "
                f"**{prediction_value:.4f} CO₂**"
            )


    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    if st.button(
        "🤖 Ask AI",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            prediction_value = None
            previous_value = None
            lag2_value = None
            lag3_value = None


            # -----------------------------------------------
            # Get prediction context
            # -----------------------------------------------

            if st.session_state.prediction:

                prediction_value = (
                    st.session_state
                    .prediction
                    .get("prediction", {})
                    .get("value")
                )


            if st.session_state.prediction_data:

                prediction_data = (
                    st.session_state
                    .prediction_data
                )

                previous_value = (
                    prediction_data.get(
                        "lag_1"
                    )
                )

                lag2_value = (
                    prediction_data.get(
                        "lag_2"
                    )
                )

                lag3_value = (
                    prediction_data.get(
                        "lag_3"
                    )
                )


            # -----------------------------------------------
            # AI request
            # -----------------------------------------------

            data = {

                "selected_subject":
                    selected_subject,

                "question":
                    question,

                "predicted_emission":
                    prediction_value,

                "previous_emission":
                    previous_value,

                "lag_2":
                    lag2_value,

                "lag_3":
                    lag3_value,

                "unit":
                    "kg CO2"

            }


            with st.spinner(
                "AI is analyzing your question..."
            ):

                result = ask_ai(
                    data
                )


            if result:

                st.session_state.ai_response = (
                    result
                )


                st.success(
                    "AI response generated!"
                )


                st.markdown(
                    "### 💬 AI Answer"
                )


                st.markdown(
                    f"""
                    <div class="ai-box">

                    {result.get("answer", "No answer returned.")}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

elif page == "📊 Prediction Analysis":

    st.title(
        "📊 Prediction Analysis"
    )


    if not st.session_state.prediction:

        st.warning(
            "No prediction available yet."
        )

        st.info(
            "Go to 'Carbon Prediction' "
            "and generate a prediction first."
        )


    else:

        result = (
            st.session_state.get("prediction")
            or {}
        )

        prediction_data = (
            st.session_state.get("prediction_data")
            or {}
        )


        prediction_value = (
            (result.get("prediction") or {})
            .get("value", 0.0)
        )


        previous = float(
            prediction_data.get("lag_1") or 0
        )


        lag2 = (
            prediction_data.get("lag_2")
        )


        lag3 = (
            prediction_data.get("lag_3")
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Predicted",
                f"{prediction_value:.4f}"
            )


        with col2:

            st.metric(
                "Previous",
                f"{previous:.4f}"
            )


        with col3:

            difference = (
                prediction_value - previous
            )

            st.metric(
                "Difference",
                f"{difference:.4f}"
            )


        with col4:

            if previous != 0:

                percentage = (
                    difference
                    / abs(previous)
                ) * 100

            else:

                percentage = 0


            st.metric(
                "Change %",
                f"{percentage:.2f}%"
            )


        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "Emission Trend"
        )


        chart_data = pd.DataFrame({

            "Period": [
                "3 Periods Ago",
                "2 Periods Ago",
                "Previous",
                "Predicted"
            ],

            "Emission": [
                lag3,
                lag2,
                previous,
                prediction_value
            ]

        })


        fig = px.line(

            chart_data,

            x="Period",

            y="Emission",

            markers=True,

            title="Historical vs Predicted CO₂"

        )


        fig.update_layout(
            xaxis_title="Period",
            yaxis_title="CO₂ Emission"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # AUTOMATIC INTERPRETATION
        # ----------------------------------------------------

        st.subheader(
            "📌 Prediction Interpretation"
        )


        if difference > 0:

            st.warning(
                f"""
                The predicted emission is higher than
                the previous emission by
                **{difference:.4f} CO₂**
                ({percentage:.2f}%).
                """
            )

        elif difference < 0:

            st.success(
                f"""
                The predicted emission is lower than
                the previous emission by
                **{abs(difference):.4f} CO₂**
                ({abs(percentage):.2f}%).
                """
            )

        else:

            st.info(
                "The predicted emission is unchanged "
                "from the previous value."
            )


        # ----------------------------------------------------
        # AI ANALYSIS BUTTON
        # ----------------------------------------------------

        st.markdown("---")


        if st.button(
            "🤖 Ask AI to Analyze This Prediction",
            use_container_width=True
        ):

            question = (
                "Analyze this prediction and explain "
                "whether emissions are increasing or "
                "decreasing."
            )


            data = {

                "selected_subject":
                    "Emission Prediction",

                "question":
                    question,

                "predicted_emission":
                    prediction_value,

                "previous_emission":
                    previous,

                "lag_2":
                    lag2,

                "lag_3":
                    lag3,

                "unit":
                    "kg CO2"

            }


            with st.spinner(
                "AI is analyzing the prediction..."
            ):

                result = ask_ai(
                    data
                )


            if result:

                st.markdown(
                    "### 🤖 AI Analysis"
                )


                st.markdown(
                    f"""
                    <div class="ai-box">

                    {result.get("answer", "")}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Supply Chain Carbon Tracker | "
    "XGBoost + Hugging Face + FastAPI + Streamlit"
)