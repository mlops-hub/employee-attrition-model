import pandas as pd
import joblib
from config.paths import MODEL_PATH, FEATURE_STORE_PATH, PREPROCESSOR_PATH

THRESHOLD_VALUE = 0.65


def test(input_data: dict):
    # Load artifacts
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURE_STORE_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Artifacts Loaded!")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    # Convert input to DataFrame with correct column order
    df_input = pd.DataFrame([input_data], columns=features)

    # Scale numeric columns
    numeric_cols = ['Years at Company', 'Company Tenure', 'RoleStagnationRatio', 'TenureGap']
    df_input[numeric_cols] = preprocessor.transform(df_input[numeric_cols])

    # Predict probability
    probability = model.predict_proba(df_input)[0][1]
    prediction = 1 if probability >= THRESHOLD_VALUE else 0

    print(f"\nAttrition Probability: {probability:.4f}")
    print(f"⌛ Prediction: {prediction}")
    print(f"Final verdict: {'😢 Leave' if 1 else '😃 Stay'}")

    return probability, prediction


if __name__ == "__main__":
    print("=" * 60)
    print("👔 Employee Attrition Prediction App")
    print("=" * 60)

    # Collect inputs
    print("Please Enter 'valid' values to get correct predicition:")
    print("-" * 60)

    years_at_company = float(input('Years at Company: '))
    performance_rating = float(input('Performance Rating (Low: 1, Below Avg: 2, Avg: 3, High: 4): '))
    no_of_promotions = int(input('Number of Promotions: '))
    overtime = int(input('Overtime (Low: 0, High: 1): '))
    edu_level = int(input('Education Level (School: 1, Bachelors-degree: 2, Master-degree: 3, Associate: 4, PhD: 5): '))
    no_of_dependents = int(input('Number of Dependents: '))
    job_level = int(input('Job Level (Entry: 1, Mid: 2, Senior: 3): '))
    company_size = int(input('Company Size (Small: 1, Medium: 2, Large: 3): '))
    company_tenure = float(input('Company Tenure: '))
    remote_work = int(input('Remote Work (No: 0, Yes: 1): '))
    company_reputation = float(input('Company Reputation (Poor: 1, Fair: 2, Good: 3, Excellent: 4): '))
    overall_satisfaction = float(input('Overall Satisfaction (Low: 1, Medium: 2, High: 3, Very High: 4): '))
    opportunities = float(input('Opportunities (Low: 0, High: 1): '))
    annual_income = int(input('Annual Income: '))
    age_group = int(input('Age Group (till 65): '))

    # Derived features
    role_stagnation_ratio = round(years_at_company / (company_tenure + 1), 3)
    tenure_gap = round(company_tenure - years_at_company, 2)
    early_company_tenure_risk = 1 if years_at_company <= 2 else 0
    long_tenure_low_role_risk = 1 if (company_tenure > 5 and job_level <= 2) else 0

    # Build input dictionary
    input_record = {
        "Years at Company": years_at_company,
        "Performance Rating": performance_rating,
        "Number of Promotions": no_of_promotions,
        "Overtime": overtime,
        "Education Level": edu_level,
        "Number of Dependents": no_of_dependents,
        "Job Level": job_level,
        "Company Size": company_size,
        "Company Tenure": company_tenure,
        "Remote Work": remote_work,
        "Company Reputation": company_reputation,
        "OverallSatisfaction": overall_satisfaction,
        "Opportunities": opportunities,
        "AnnualIncome": annual_income,
        "AgeGroup": age_group,
        "RoleStagnationRatio": role_stagnation_ratio,
        "TenureGap": tenure_gap,
        "EarlyCompanyTenureRisk": early_company_tenure_risk,
        "LongTenureLowRoleRisk": long_tenure_low_role_risk
    }

    # Run test
    test(input_record)
