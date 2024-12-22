import pickle
import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import BytesIO

# Load the trained model from GitHub
@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/Arnob83/Deploy-model/main/XGBoost_model.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        raise FileNotFoundError(f"Failed to download the model file. Status code: {response.status_code}")

classifier = load_model()

@st.cache_data
def prediction(Education_1, ApplicantIncome, CoapplicantIncome, Credit_History, Loan_Amount_Term):
    # Convert user input
    Education_1 = 0 if Education_1 == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data in the expected order
    input_data = pd.DataFrame(
        [[Education_1, ApplicantIncome, CoapplicantIncome, Credit_History, Loan_Amount_Term]],
        columns=["Education_1", "ApplicantIncome", "CoapplicantIncome", "Credit_History", "Loan_Amount_Term"]
    )

    # Ensure column order matches the classifier’s expectations
    input_data = input_data[classifier.feature_names_in_]

    # Model prediction (0 = Rejected, 1 = Approved)
    prediction = classifier.predict(input_data)
    if prediction[0] == 0:
        pred_label = 'Rejected'
    else:
        pred_label = 'Approved'
    return pred_label, input_data

def explain_with_bar_chart_and_text(input_data, final_result):
    """
    Generates a SHAP bar chart and text explanation.
    """
    # Initialize SHAP explainer
    explainer = shap.Explainer(classifier)
    shap_values = explainer(input_data)

    shap_values_for_first_sample = shap_values.values[0]
    predicted_class = 1 if final_result == 'Approved' else 0
    if shap_values_for_first_sample.ndim > 1:
        shap_values_for_first_sample = shap_values_for_first_sample[:, predicted_class]

    # Prepare data for plotting and text explanation
    feature_names = input_data.columns
    contributions = dict(zip(feature_names, shap_values_for_first_sample))

    # --- Create bar chart ---
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_first_sample, color="skyblue")
    plt.title("Feature Contributions to Prediction")
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.tight_layout()

    # --- Generate a text explanation ---
    explanation_text = "Explanation of Features and Their SHAP Contributions:\n\n"
    for feature, shap_value in contributions.items():
        actual_value = input_data[feature].iloc[0]
        if shap_value > 0:
            explanation_text += (
                f"- **{feature}** = {actual_value}, positively influenced (pushed) towards approval "
                f"(SHAP: {shap_value:.2f}).\n"
            )
        elif shap_value < 0:
            explanation_text += (
                f"- **{feature}** = {actual_value}, negatively influenced (pushed) towards rejection "
                f"(SHAP: {shap_value:.2f}).\n"
            )
        else:
            explanation_text += (
                f"- **{feature}** = {actual_value}, no significant impact (SHAP: 0).\n"
            )

    # Most Influential Feature
    largest_contributor_feature = max(contributions, key=lambda k: abs(contributions[k]))
    largest_contributor_value = contributions[largest_contributor_feature]
    explanation_text += (
        f"\n**Most Influential Feature**: {largest_contributor_feature} "
        f"(SHAP = {largest_contributor_value:.2f})."
    )

    return plt, explanation_text

def main():
    # Front-end elements
    st.markdown(
        """
        <div style="background-color:Yellow;padding:13px">
        <h1 style="color:black;text-align:center;">Loan Prediction ML App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User inputs
    Education_1 = st.selectbox('Education', ("Under_Graduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's Monthly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's Monthly Income", min_value=0.0)
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    # Prediction
    if st.button("Predict"):
        result, input_data = prediction(
            Education_1,
            ApplicantIncome,
            CoapplicantIncome,
            Credit_History,
            Loan_Amount_Term
        )

        # Show result in color
        if result == "Approved":
            st.success(f'Your loan is {result}', icon="✅")
        else:
            st.error(f'Your loan is {result}', icon="❌")

        # Explanation (with SHAP)
        st.header("Explanation of Prediction")
        bar_chart, explanation_text = explain_with_bar_chart_and_text(input_data, final_result=result)
        st.pyplot(bar_chart)
        st.write(explanation_text)

if __name__ == '__main__':
    main()
