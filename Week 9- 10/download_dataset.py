import os
import numpy as np
import pandas as pd

def prepare_dataset():
    """Generates a realistic Financial Credit Underwriting & Loan Default Dataset (10,000 applicants)
    with intentional missing data patterns, heavy right-skewed distributions, and extreme debt outliers.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'loan_underwriting.csv')

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100000:
        print(f"  [+] Dataset already exists: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
        return csv_path

    print("  [+] Generating Financial Underwriting & Credit Risk Dataset (10,000 applicants)...")
    np.random.seed(42)
    n_samples = 10000

    # 1. Base Continuous Features with realistic skewed/normal shapes
    # Annual income: log-normal distribution (right-skewed, typical of salaries)
    log_income = np.random.normal(loc=11.1, scale=0.55, size=n_samples)
    annual_income = np.clip(np.exp(log_income), 18000, 320000).round(-2)

    # Credit Score (FICO range 300 - 850)
    credit_score = np.clip(np.random.normal(loc=675, scale=55, size=n_samples), 480, 830).astype(int)

    # Employment Duration (Years)
    emp_years = np.clip(np.random.exponential(scale=5.0, size=n_samples), 0, 35).round(1)

    # Loan Amount Requested ($1,000 - $40,000)
    loan_amount = np.random.choice(
        np.arange(3000, 41000, 500),
        size=n_samples,
        p=None
    )

    # Interest rate depends negatively on credit score
    base_rate = 26.0 - (credit_score - 480) * 0.045 + np.random.normal(0, 1.8, size=n_samples)
    interest_rate = np.clip(base_rate, 5.2, 29.5).round(2)

    # Revolving Balance (Credit card debt - heavily right-skewed with extreme outliers)
    raw_balance = np.exp(np.random.normal(loc=8.8, scale=1.1, size=n_samples))
    revolving_balance = np.clip(raw_balance, 300, 145000).round(2)

    # Revolving Credit Utilization Ratio (0.0 to 1.25)
    revolving_utilization = np.clip(np.random.beta(a=2.5, b=3.0, size=n_samples) * 1.15, 0.02, 1.20).round(3)

    # Total open credit lines (2 to 45)
    total_credit_lines = np.clip(np.random.poisson(lam=12, size=n_samples), 2, 48)

    # Delinquencies in the past 2 years (0, 1, 2, 3+)
    delinquencies = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.82, 0.11, 0.04, 0.02, 0.01])

    # 2. Categorical Features
    home_ownership = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], size=n_samples, p=[0.48, 0.41, 0.11])
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase'],
        size=n_samples,
        p=[0.52, 0.24, 0.10, 0.08, 0.06]
    )
    application_type = np.random.choice(['INDIVIDUAL', 'JOINT'], size=n_samples, p=[0.86, 0.14])

    # 3. Ground Truth Default Probability Logic (Log-Odds formulation)
    # Risk factors: low credit score, high utilization, high interest rate, high debt relative to income, delinquencies
    dti_ratio = (loan_amount * 0.35 + revolving_balance * 0.15) / annual_income
    
    log_odds = (
        -1.8
        - 0.014 * (credit_score - 675)
        + 0.12 * (interest_rate - 12.0)
        + 2.2 * (revolving_utilization - 0.45)
        + 3.5 * (dti_ratio - 0.18)
        + 0.45 * delinquencies
        + (0.35 if 'RENT' else 0.0)
        + np.random.normal(0, 0.5, size=n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-log_odds))
    # Generate binary default label (~21% prevalence)
    is_default = (np.random.rand(n_samples) < prob_default).astype(int)

    # 4. Inject Realistic Missing Data (MCAR & MAR)
    # Annual income missing in 6.2% of records (e.g. self-employed / unverified)
    income_missing_idx = np.random.choice(n_samples, size=int(0.062 * n_samples), replace=False)
    annual_income_with_nan = annual_income.astype(float)
    annual_income_with_nan[income_missing_idx] = np.nan

    # Credit score missing in 7.8% of records (thin-file / international applicants)
    score_missing_idx = np.random.choice(n_samples, size=int(0.078 * n_samples), replace=False)
    credit_score_with_nan = credit_score.astype(float)
    credit_score_with_nan[score_missing_idx] = np.nan

    # Employment years missing in 5.4% of records (students, retirees, unstated)
    emp_missing_idx = np.random.choice(n_samples, size=int(0.054 * n_samples), replace=False)
    emp_years_with_nan = emp_years.astype(float)
    emp_years_with_nan[emp_missing_idx] = np.nan

    # Home ownership missing in 3.1% of records (unspecified in form)
    home_missing_idx = np.random.choice(n_samples, size=int(0.031 * n_samples), replace=False)
    home_ownership_with_nan = home_ownership.astype(object)
    home_ownership_with_nan[home_missing_idx] = np.nan

    # Assemble DataFrame
    df = pd.DataFrame({
        'annual_income': annual_income_with_nan,
        'credit_score': credit_score_with_nan,
        'employment_years': emp_years_with_nan,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'revolving_balance': revolving_balance,
        'revolving_utilization': revolving_utilization,
        'total_credit_lines': total_credit_lines,
        'delinquencies_2yrs': delinquencies,
        'home_ownership': home_ownership_with_nan,
        'loan_purpose': loan_purpose,
        'application_type': application_type,
        'is_default': is_default
    })

    df.to_csv(csv_path, index=False)
    print(f"  [+] Saved {len(df):,} loan applicant records to: {csv_path}")
    return csv_path

if __name__ == '__main__':
    prepare_dataset()
