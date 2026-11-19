
import streamlit as st
import pandas as pd
import json

from pathlib import Path

from src.data_loader import BASE_DIR
from src.triage_agent import run_triage
from src.maintenance.work_order import generate_work_order

from mlops.monitoring import (
    get_prediction_logs,
    get_monitoring_summary,
    check_model_health
)


# ==================================================
# 1. PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="NeuroMaintain | AI Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #CBD5E1;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 0px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 4px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 15px;
}

.info-card {
    background: linear-gradient(135deg, #17243A, #111C2F);
    border: 1px solid #263B57;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
}

.card-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
}

.card-value {
    color: #F8FAFC;
    font-size: 25px;
    font-weight: 750;
    margin-top: 10px;
}

.card-description {
    color: #64748B;
    font-size: 12px;
    margin-top: 5px;
}

.status-box {
    background-color: #14243A;
    border-left: 4px solid #06B6D4;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0px;
}

div.stButton > button {
    background: linear-gradient(90deg, #0891B2, #2563EB);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 650;
    padding: 10px 18px;
    transition: 0.2s;
}

div.stButton > button:hover {
    border: 1px solid #38BDF8;
    color: white;
    transform: translateY(-2px);
}

div[data-testid="stMetric"] {
    background-color: #142137;
    border: 1px solid #263B57;
    padding: 17px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #CBD5E1;
}

hr {
    border-color: #263449;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE INITIALIZATION
# ==================================================

defaults = {
    "result": None,
    "work_order": None,
    "sensor_data": None,
    "shap_result": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def render_card(label, value, description):

    st.markdown(f"""
    <div class="info-card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_metrics(result):

    probability = float(result["failure_probability"]) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Failure Probability",
            f"{probability:.2f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    with col3:
        st.metric(
            "Anomaly Status",
            "Detected" if result["anomaly_detected"] else "Normal"
        )

    with col4:
        st.metric(
            "Prediction",
            "Failure" if result["prediction"] == 1 else "No Failure"
        )


def render_header():

    st.markdown(
        '<div class="main-title">⚙️ NeuroMaintain</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Industrial Predictive Maintenance Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")


def get_summary():

    return get_monitoring_summary()


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown("## ⚙️ NeuroMaintain")

    st.caption("INDUSTRIAL AI INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "🔍 Machine Analysis",
            "🧠 AI Explainability",
            "🔧 Maintenance",
            "📊 MLOps Monitoring"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### System Information")

    st.success("AI Pipeline Connected")

    st.caption("XGBoost | PyTorch | SHAP")

    st.caption("Predictive Maintenance System")

    st.divider()

    st.caption("NeuroMaintain v1.0")


# ==================================================
# 6. OVERVIEW PAGE
# ==================================================

render_header()

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Monitor machine health, failure risks and AI-powered maintenance."
    )

    try:
        summary = get_summary()

        col1, col2, col3 = st.columns(3)

        with col1:
            render_card(
                "TOTAL PREDICTIONS",
                summary["total_predictions"],
                "Total machine analyses"
            )

        with col2:
            render_card(
                "AVERAGE FAILURE RISK",
                f'{summary["average_failure_probability"] * 100:.2f}%',
                "Average predicted failure probability"
            )

        with col3:
            render_card(
                "ANOMALIES DETECTED",
                summary["anomaly_count"],
                "Detected by anomaly detection model"
            )

    except Exception as error:
        st.warning(f"Monitoring summary unavailable: {error}")

    st.write("")
    st.divider()

    st.markdown("### Latest Machine Analysis")

    result = st.session_state["result"]

    if result is not None:

        render_prediction_metrics(result)

        st.markdown(
            f"""
            <div class="status-box">
                <b>Recommended Action</b><br><br>
                {result["recommended_action"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f'Inference Time: {result["inference_time_seconds"]} seconds'
        )

    else:

        st.info(
            "No recent analysis available. "
            "Open Machine Analysis to analyze a machine."
        )

    st.divider()

    st.markdown("### Quick Access")

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("**🔍 Machine Analysis**")
        st.write("Enter sensor readings and predict machine failure.")

    with q2:
        st.markdown("**🧠 AI Explainability**")
        st.write("Understand the factors influencing predictions.")


# ==================================================
# 7. MACHINE ANALYSIS PAGE
# ==================================================

elif page == "🔍 Machine Analysis":

    st.markdown(
        '<div class="section-title">Machine Health Analysis</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter industrial sensor readings to analyze machine health."
    )

    with st.container(border=True):

        st.markdown("### Machine Configuration")

        c1, c2 = st.columns(2)

        with c1:
            machine_id = st.text_input(
                "Machine ID",
                value="MACHINE_001"
            )

        with c2:
            machine_type = st.selectbox(
                "Machine Type",
                ["L", "M", "H"]
            )

        st.divider()

        st.markdown("### Sensor Parameters")

        st.caption("Provide the current machine sensor readings.")

        col1, col2, col3 = st.columns(3)

        with col1:

            air_temp = st.number_input(
                "Air Temperature (K)",
                value=300.5,
                step=0.1
            )

            process_temp = st.number_input(
                "Process Temperature (K)",
                value=310.0,
                step=0.1
            )

        with col2:

            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                value=1500,
                step=1
            )

            torque = st.number_input(
                "Torque (Nm)",
                value=45.0,
                step=0.1
            )

        with col3:

            tool_wear = st.number_input(
                "Tool Wear (min)",
                value=120,
                step=1
            )

        sensor_data = {

            "Machine_ID": machine_id,

            "Type": machine_type,

            "Air temperature [K]": air_temp,

            "Process temperature [K]": process_temp,

            "Rotational speed [rpm]": rotational_speed,

            "Torque [Nm]": torque,

            "Tool wear [min]": tool_wear

        }

        st.divider()

        analyze = st.button(
            "🚀 Analyze Machine Health",
            type="primary",
            use_container_width=True
        )

    if analyze:

        if not machine_id.strip():

            st.error("Please enter a valid Machine ID.")

        else:

            try:

                with st.spinner(
                    "Running AI prediction and anomaly detection..."
                ):

                    result = run_triage(sensor_data)

                    work_order = generate_work_order(
                        result,
                        sensor_data
                    )

                    st.session_state["result"] = result

                    st.session_state["work_order"] = work_order

                    st.session_state["sensor_data"] = sensor_data

                    st.session_state["shap_result"] = None

                st.success("Machine analysis completed successfully!")

            except Exception as error:

                st.error(f"Analysis failed: {error}")

    # DISPLAY ANALYSIS RESULT

    result = st.session_state["result"]

    if result is not None:

        st.divider()

        st.markdown("### Analysis Results")

        tab1, tab2, tab3 = st.tabs([
            "📌 Prediction",
            "🔎 AI Insights",
            "🔧 Recommendation"
        ])

        with tab1:

            render_prediction_metrics(result)

            st.write("")

            st.metric(
                "Inference Time",
                f'{result["inference_time_seconds"]:.4f} sec'
            )

        with tab2:

            st.markdown("### Anomaly Detection")

            if result["anomaly_detected"]:

                st.error("Anomaly detected in machine sensor readings.")

            else:

                st.success("No anomaly detected.")

            st.markdown("### Failure Classification")

            st.write(
                "Prediction:",
                "Failure" if result["prediction"] == 1 else "No Failure"
            )

            st.write(
                "Risk Level:",
                result["risk_level"]
            )

        with tab3:

            st.markdown("### Recommended Maintenance Action")

            st.warning(result["recommended_action"])

            st.write(
                "**Maintenance Priority:**",
                result["maintenance_priority"]
            )

        st.caption(
            "For additional details, visit AI Explainability "
            "and Maintenance from the sidebar."
        )


# ==================================================
# 8. AI EXPLAINABILITY PAGE
# ==================================================

elif page == "🧠 AI Explainability":

    st.markdown(
        '<div class="section-title">AI Explainability Center</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Understand why the AI model predicted a particular outcome."
    )

    if st.session_state["result"] is None:

        st.info("Analyze a machine first to generate an explanation.")

    else:

        result = st.session_state["result"]

        render_prediction_metrics(result)

        st.divider()

        st.markdown("### SHAP Feature Explanation")

        st.write(
            "SHAP helps identify which sensor parameters "
            "contributed to the model prediction."
        )

        if st.button(
            "🧠 Generate SHAP Explanation",
            type="primary",
            use_container_width=True
        ):

            try:

                from src.explainability.shap_explainer import (
                    explain_prediction
                )

                with st.spinner("Generating SHAP explanation..."):

                    shap_result = explain_prediction(
                        st.session_state["sensor_data"]
                    )

                    st.session_state["shap_result"] = shap_result

            except Exception as error:

                st.error(f"SHAP explanation failed: {error}")

        shap_result = st.session_state["shap_result"]

        if shap_result is not None:

            st.divider()

            st.markdown("### Top Contributing Features")

            st.dataframe(
                shap_result["top_features"],
                use_container_width=True,
                hide_index=True
            )

            plot_path = Path(shap_result["plot_path"])

            if plot_path.exists():

                st.markdown("### SHAP Visualization")

                st.image(
                    str(plot_path),
                    use_container_width=True
                )

            else:

                st.warning("SHAP plot file was not found.")


# ==================================================
# 9. MAINTENANCE PAGE
# ==================================================

elif page == "🔧 Maintenance":

    st.markdown(
        '<div class="section-title">Maintenance Management</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI-generated maintenance recommendations and work orders."
    )

    work_order = st.session_state["work_order"]

    if work_order is None:

        st.info(
            "No maintenance work order available. "
            "Analyze a machine first."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            render_card(
                "WORK ORDER ID",
                work_order["work_order_id"],
                "Generated automatically"
            )

        with col2:
            render_card(
                "MACHINE ID",
                work_order["machine_id"],
                "Target machine"
            )

        with col3:
            render_card(
                "PRIORITY",
                work_order["priority"],
                "Maintenance priority"
            )

        st.write("")

        st.markdown("### Work Order Details")

        st.write("**Status:**", work_order["status"])

        st.write("**Priority:**", work_order["priority"])

        st.divider()

        st.markdown("### Maintenance Instructions")

        for index, instruction in enumerate(
            work_order["instructions"], start=1
        ):

            st.markdown(f"**{index}.** {instruction}")

        st.divider()

        st.download_button(
            "⬇️ Download Maintenance Work Order",
            data=json.dumps(work_order, indent=4),
            file_name=f'{work_order["work_order_id"]}.json',
            mime="application/json",
            use_container_width=True
        )


# ==================================================
# 10. MLOPS MONITORING PAGE
# ==================================================

elif page == "📊 MLOps Monitoring":

    st.markdown(
        '<div class="section-title">MLOps Monitoring Center</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Track model availability, prediction statistics and logs."
    )

    try:

        summary = get_monitoring_summary()

        st.markdown("### Prediction Statistics")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total Predictions",
                summary["total_predictions"]
            )

        with c2:
            st.metric(
                "Average Failure Probability",
                f'{summary["average_failure_probability"] * 100:.2f}%'
            )

        with c3:
            st.metric(
                "Anomalies Detected",
                summary["anomaly_count"]
            )

        st.divider()

        st.markdown("### Risk Distribution")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Critical", summary["critical_count"])

        with c2:
            st.metric("High", summary["high_count"])

        with c3:
            st.metric("Medium", summary["medium_count"])

        with c4:
            st.metric("Low", summary["low_count"])

        # Risk distribution chart

        risk_data = pd.DataFrame({
            "Risk Level": ["Critical", "High", "Medium", "Low"],
            "Count": [
                summary["critical_count"],
                summary["high_count"],
                summary["medium_count"],
                summary["low_count"]
            ]
        })

        st.bar_chart(
            risk_data.set_index("Risk Level"),
            use_container_width=True
        )

    except Exception as error:

        st.error(f"Monitoring summary unavailable: {error}")

    st.divider()

    # MODEL HEALTH

    st.markdown("### Model Health Status")

    try:

        health = check_model_health()

        cols = st.columns(max(1, len(health)))

        for index, (model_name, status) in enumerate(health.items()):

            with cols[index]:

                if status == "AVAILABLE":

                    st.success(
                        f"🟢 {model_name}\n\n{status}"
                    )

                else:

                    st.error(
                        f"🔴 {model_name}\n\n{status}"
                    )

    except Exception as error:

        st.error(f"Model health check failed: {error}")

    st.divider()

    # PREDICTION HISTORY

    st.markdown("### Prediction History")

    try:

        logs = get_prediction_logs()

        if not logs.empty:

            display_logs = logs.drop(
                columns=["sensor_data"],
                errors="ignore"
            ).tail(20)

            st.dataframe(
                display_logs,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Prediction Logs",
                data=logs.to_csv(index=False),
                file_name="neuromaintain_prediction_logs.csv",
                mime="text/csv"
            )

        else:

            st.info("No prediction records available yet.")

    except Exception as error:

        st.error(f"Unable to load prediction logs: {error}")


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:15px;">
        <b>NeuroMaintain</b> | Explainable AI-Based Predictive Maintenance
        <br><br>
        Built with Python | PyTorch | XGBoost | SHAP | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
