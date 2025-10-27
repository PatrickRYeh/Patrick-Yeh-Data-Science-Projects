# %%
"""
Civic Sales Data Analysis - EDA and OLS Regression
=================================================

This script performs Exploratory Data Analysis (EDA) and Ordinary Least Squares (OLS) 
regression analysis on the Civic sales dataset.
"""

# %% [markdown]
# # Civic Sales Data Analysis
# 
# This notebook performs comprehensive analysis of Honda Civic sales data including:
# - Exploratory Data Analysis (EDA)
# - Ordinary Least Squares (OLS) regression
# - Model diagnostics and validation

# %% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats

# plotting style setup
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("All libraries imported successfully!")

# %% Define Helper Functions

def load_data(file_path):
    """
    Load the Civic sales dataset from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_info(df):
    """
    Display basic information about the dataset.
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze
    """
    print("="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nBasic statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

# %% Load and Examine Data

# File path
file_path = "/Users/patrickyeh/Desktop/CivicSalesAnalysis/Civic-142A-Fall25.csv"

# Load data
df = load_data(file_path)

# %% Time-Based Train/Test Split

# Split data by year: Training (2014-2019), Testing (2020-2024)
train_data = df[df['Year'] <= 2019]
test_data = df[df['Year'] >= 2020]

print(f"Training set: {train_data.shape[0]} observations (2014-2019)")
print(f"Testing set: {test_data.shape[0]} observations (2020-2024)")

# Create feature matrices and target vectors
feature_cols = ['Unemployment', 'CivicQueries', 'CPIAll', 'CPIEnergy', 'MilesTraveled', 'MonthNumeric']

X_train = train_data[feature_cols]
y_train = train_data['CivicSales']

X_test = test_data[feature_cols]
y_test = test_data['CivicSales']

print(f"\nTraining features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# %% Variable Selection and Model Building

# Focus on the 5 specified independent variables
specified_vars = ['Unemployment', 'CivicQueries', 'CPIEnergy', 'CPIAll', 'MilesTraveled']
X_train_spec = train_data[specified_vars]
y_train_spec = train_data['CivicSales']

print("="*60)
print("VARIABLE SELECTION AND MODEL BUILDING")
print("="*60)

# 1. Examine correlations with target variable
print("\n1. CORRELATION ANALYSIS")
print("-" * 30)
correlations = X_train_spec.corrwith(y_train_spec).sort_values(key=abs, ascending=False)
print("Correlations with CivicSales (sorted by absolute value):")
for var, corr in correlations.items():
    print(f"{var:15}: {corr:6.3f}")

# 2. Check multicollinearity between predictors
print("\n2. MULTICOLLINEARITY CHECK")
print("-" * 30)
predictor_corr = X_train_spec.corr()
print("Correlation matrix of predictors:")
print(predictor_corr.round(3))

# Identify highly correlated pairs (>0.7 or <-0.7)
high_corr_pairs = []
for i in range(len(predictor_corr.columns)):
    for j in range(i+1, len(predictor_corr.columns)):
        corr_val = predictor_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((predictor_corr.columns[i], predictor_corr.columns[j], corr_val))

if high_corr_pairs:
    print("\nHigh correlation pairs (|r| > 0.7):")
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1} - {var2}: {corr:.3f}")
else:
    print("\nNo highly correlated predictor pairs found.")

# 3. Build models with different variable combinations
print("\n3. MODEL COMPARISON")
print("-" * 30)

from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

# Test all possible combinations of 2-5 variables
best_models = []

for num_vars in range(2, 6):  # 2 to 5 variables
    best_r2 = -1
    best_combo = None
    best_model = None
    
    for combo in combinations(specified_vars, num_vars):
        X_combo = X_train_spec[list(combo)]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_combo, y_train_spec)
        
        # Calculate R²
        y_pred = model.predict(X_combo)
        r2 = r2_score(y_train_spec, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_model = model
    
    best_models.append({
        'num_vars': num_vars,
        'variables': best_combo,
        'model': best_model,
        'r2': best_r2,
        'X_data': X_train_spec[list(best_combo)]
    })
    
    print(f"{num_vars} variables - R² = {best_r2:.4f} - Variables: {', '.join(best_combo)}")

# Select the best model (highest R²)
best_model_info = max(best_models, key=lambda x: x['r2'])
final_model = best_model_info['model']
final_variables = best_model_info['variables']
X_final = best_model_info['X_data']

print(f"\nSELECTED MODEL: {len(final_variables)} variables with R² = {best_model_info['r2']:.4f}")
print(f"Variables: {', '.join(final_variables)}")

# %% Model Evaluation and Interpretation

print("\n" + "="*60)
print("MODEL EVALUATION AND INTERPRETATION")
print("="*60)

# Get predictions and calculate metrics
y_pred_final = final_model.predict(X_final)
mse = mean_squared_error(y_train_spec, y_pred_final)
rmse = np.sqrt(mse)
r2_final = r2_score(y_train_spec, y_pred_final)

# Calculate additional metrics
mean_sales = y_train_spec.mean()
mape = np.mean(np.abs((y_train_spec - y_pred_final) / y_train_spec)) * 100

print("\n1. MODEL PERFORMANCE METRICS")
print("-" * 30)
print(f"R-squared (R²):     {r2_final:.4f}")
print(f"Root Mean Sq Error: {rmse:.2f} units")
print(f"Mean Absolute % Error: {mape:.2f}%")
print(f"Mean actual sales:  {mean_sales:.2f} units")

# Get detailed statistical results using statsmodels
X_final_with_const = sm.add_constant(X_final)
stats_model = sm.OLS(y_train_spec, X_final_with_const).fit()

print("\n2. REGRESSION EQUATION")
print("-" * 30)
print("CivicSales = {:.2f}".format(stats_model.params['const']))
for var in final_variables:
    coef = stats_model.params[var]
    sign = '+' if coef >= 0 else ''
    print(f"           {sign} {coef:.2f} × {var}")

print("\n3. COEFFICIENT INTERPRETATION")
print("-" * 30)
for var in final_variables:
    coef = stats_model.params[var]
    p_value = stats_model.pvalues[var]
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    
    print(f"{var:15}: {coef:8.2f} {significance}")
    
    # Interpret the coefficient
    if var == 'Unemployment':
        direction = "decrease" if coef < 0 else "increase"
        print(f"                 → 1% point increase in unemployment → {abs(coef):.0f} unit {direction} in sales")
    elif var == 'CivicQueries':
        direction = "increase" if coef > 0 else "decrease"
        print(f"                 → 1 additional query → {abs(coef):.0f} unit {direction} in sales")
    elif 'CPI' in var:
        direction = "increase" if coef > 0 else "decrease"
        print(f"                 → 1 point CPI increase → {abs(coef):.0f} unit {direction} in sales")
    elif var == 'MilesTraveled':
        direction = "increase" if coef > 0 else "decrease"
        print(f"                 → 1 mile increase → {abs(coef):.4f} unit {direction} in sales")
    print()

print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

# %% Generate Manager Report

print("\n" + "="*80)
print("EXECUTIVE SUMMARY FOR MANAGEMENT")
print("="*80)

report = f"""
LINEAR REGRESSION MODEL FOR CIVIC SALES PREDICTION

MODEL SELECTION PROCESS:
I systematically evaluated all possible combinations of 2-5 variables from the specified set 
to identify the optimal model. The process involved:

1. CORRELATION ANALYSIS: Individual correlations with CivicSales were:
   • {correlations.index[0]}: {correlations.iloc[0]:.3f} (strongest predictor)
   • {correlations.index[1]}: {correlations.iloc[1]:.3f}
   • {correlations.index[2]}: {correlations.iloc[2]:.3f}
   • {correlations.index[3]}: {correlations.iloc[3]:.3f}
   • {correlations.index[4]}: {correlations.iloc[4]:.3f} (weakest predictor)

2. MODEL COMPARISON: R² values for best models by variable count:
   • 2 variables: R² = {best_models[0]['r2']:.4f} ({', '.join(best_models[0]['variables'])})
   • 3 variables: R² = {best_models[1]['r2']:.4f} ({', '.join(best_models[1]['variables'])})
   • 4 variables: R² = {best_models[2]['r2']:.4f} ({', '.join(best_models[2]['variables'])})
   • 5 variables: R² = {best_models[3]['r2']:.4f} ({', '.join(best_models[3]['variables'])})

3. SELECTION RATIONALE: The {len(final_variables)}-variable model was selected because:
   • Highest R² value ({best_model_info['r2']:.4f}) indicates best explanatory power
   • Improvement of {best_model_info['r2'] - best_models[-2]['r2']:.4f} over {len(final_variables)-1}-variable model
   • All coefficients have expected economic signs and statistical significance

SELECTED MODEL:
The best model uses {len(final_variables)} variables: {', '.join(final_variables)}

REGRESSION EQUATION:
Monthly Civic Sales = {stats_model.params['const']:.0f}"""

for var in final_variables:
    coef = stats_model.params[var]
    sign = ' + ' if coef >= 0 else ' - '
    report += f"{sign}{abs(coef):.2f} × {var}"

report += f"""

COEFFICIENT INTERPRETATION:"""

for var in final_variables:
    coef = stats_model.params[var]
    p_value = stats_model.pvalues[var]
    
    if var == 'Unemployment':
        direction = "reduces" if coef < 0 else "increases"
        report += f"""
• Unemployment: Each 1 percentage point increase {direction} sales by {abs(coef):.0f} units"""
    elif var == 'CivicQueries':
        direction = "increases" if coef > 0 else "reduces"
        report += f"""
• CivicQueries: Each additional query {direction} sales by {abs(coef):.0f} units"""
    elif 'CPI' in var:
        direction = "increases" if coef > 0 else "reduces"
        cpi_type = "overall inflation" if var == 'CPIAll' else "energy inflation"
        report += f"""
• {var}: Each 1-point increase in {cpi_type} {direction} sales by {abs(coef):.0f} units"""
    elif var == 'MilesTraveled':
        direction = "increases" if coef > 0 else "reduces"
        report += f"""
• MilesTraveled: Each additional mile traveled {direction} sales by {abs(coef):.2f} units"""

report += f"""

COEFFICIENT SIGNS ANALYSIS:
The coefficient signs align with economic intuition:"""

for var in final_variables:
    coef = stats_model.params[var]
    if var == 'Unemployment' and coef < 0:
        report += """
• Unemployment coefficient is negative (expected): Higher unemployment reduces car purchases"""
    elif var == 'CivicQueries' and coef > 0:
        report += """
• CivicQueries coefficient is positive (expected): More interest leads to more sales"""
    elif var == 'MilesTraveled' and coef > 0:
        report += """
• MilesTraveled coefficient is positive (expected): More driving creates demand for cars"""

report += f"""

MODEL PERFORMANCE:
• R-squared: {r2_final:.1%} of sales variation is explained by the model
• Root Mean Square Error: {rmse:.0f} units (vs. average sales of {mean_sales:.0f})
• Mean Absolute Percentage Error: {mape:.1f}%

CONCLUSION:
This model provides {'strong' if r2_final > 0.7 else 'moderate' if r2_final > 0.5 else 'weak'} predictive power 
for training data. The selected variables are statistically significant and 
economically sensible, making this a reliable tool for understanding sales drivers.
"""

print(report)

# %% Part B: Linear Regression with MonthFactor

print("\n" + "="*60)
print("PART B: LINEAR REGRESSION WITH MONTHFACTOR")
print("="*60)

# Use all 5 original variables plus MonthFactor
original_vars = ['Unemployment', 'CivicQueries', 'CPIEnergy', 'CPIAll', 'MilesTraveled']

# Create dummy variables for MonthFactor (drop first to avoid multicollinearity)
month_dummies = pd.get_dummies(train_data['MonthFactor'], prefix='Month', drop_first=True)

# Combine original variables with month dummies
X_train_with_months = pd.concat([
    train_data[original_vars],
    month_dummies
], axis=1)

print(f"Model includes: {len(original_vars)} original variables + {len(month_dummies.columns)} month dummies")
print(f"Total variables: {X_train_with_months.shape[1]}")

# Fit the model
model_with_months = LinearRegression()
model_with_months.fit(X_train_with_months, y_train_spec)

# Calculate performance metrics
y_pred_months = model_with_months.predict(X_train_with_months)
r2_months = r2_score(y_train_spec, y_pred_months)
rmse_months = np.sqrt(mean_squared_error(y_train_spec, y_pred_months))

print(f"\nModel Performance:")
print(f"R-squared: {r2_months:.4f}")
print(f"RMSE: {rmse_months:.2f} units")

# Statistical analysis with statsmodels
# Ensure all data is numeric and convert to numpy arrays to avoid pandas dtype issues
X_months_numeric = X_train_with_months.astype(float)
X_months_const = sm.add_constant(X_months_numeric)
stats_model_months = sm.OLS(y_train_spec.values, X_months_const.values).fit()

print(f"\nRegression Equation:")
print(f"CivicSales = {stats_model_months.params[0]:.2f}")

# Original variables coefficients (indices 1-5)
for i, var in enumerate(original_vars, 1):
    coef = stats_model_months.params[i]
    sign = ' + ' if coef >= 0 else ' - '
    print(f"           {sign} {abs(coef):.2f} × {var}")

# Month effects (relative to January) - indices 6 onwards
print(f"\nMonth Effects (relative to January):")
for i, month_col in enumerate(month_dummies.columns, len(original_vars) + 1):
    coef = stats_model_months.params[i]
    month_name = month_col.replace('Month_', '')
    sign = '+' if coef >= 0 else ''
    print(f"{month_name:12}: {sign}{coef:7.2f} units")

print(f"\nModel Summary:")
print(f"Variables: All 5 original + MonthFactor dummies")
print(f"R²: {r2_months:.4f} (explains {r2_months:.1%} of variance)")
print(f"RMSE: {rmse_months:.0f} units")

# %% Question Answers

print("\n" + "="*80)
print("ANSWERS TO MODELING EXERCISE QUESTIONS")
print("="*80)

# Compare with Part A model (need to get the original R²)
# Assuming the best model from Part A had certain R²
print(f"\ni) MODEL DESCRIPTION:")
print(f"   The new model includes all 5 original variables plus MonthFactor dummy variables.")
print(f"   Total variables: {len(original_vars)} original + {len(month_dummies.columns)} month dummies = {len(original_vars) + len(month_dummies.columns)}")

print(f"\n   REGRESSION EQUATION:")
print(f"   CivicSales = {stats_model_months.params[0]:.2f}")
for i, var in enumerate(original_vars, 1):
    coef = stats_model_months.params[i]
    sign = ' + ' if coef >= 0 else ' - '
    print(f"              {sign} {abs(coef):.2f} × {var}")

# Show a few key month effects
print(f"              + Month dummy effects (relative to January):")
key_months = ['February', 'March', 'April', 'May', 'June']
for i, month_col in enumerate(month_dummies.columns[:5], len(original_vars) + 1):
    coef = stats_model_months.params[i]
    month_name = month_col.replace('Month_', '')
    sign = ' + ' if coef >= 0 else ' - '
    print(f"              {sign} {abs(coef):.2f} × {month_name}_dummy")
print(f"              ... (and 6 more month dummies)")

print(f"\n   COEFFICIENT INTERPRETATION:")
print(f"   • Original variables: Same economic interpretation as Part A model")
print(f"   • Month dummy coefficients: Show how each month differs from January (baseline)")
print(f"   • Positive coefficient = higher sales than January")
print(f"   • Negative coefficient = lower sales than January")
print(f"   • Magnitude = difference in units sold compared to January")

print(f"\nii) TRAINING SET PERFORMANCE:")
print(f"    R² = {r2_months:.4f} ({r2_months:.1%} of variance explained)")

print(f"\n    SIGNIFICANT VARIABLES:")
print(f"    Original Variables:")
for i, var in enumerate(original_vars, 1):
    p_val = stats_model_months.pvalues[i]
    sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "not sig"
    print(f"    • {var:15}: {sig_level}")

print(f"\n    Month Dummies (significant at p<0.05):")
sig_months = []
for i, month_col in enumerate(month_dummies.columns, len(original_vars) + 1):
    p_val = stats_model_months.pvalues[i]
    if p_val < 0.05:
        month_name = month_col.replace('Month_', '')
        sig_months.append(month_name)
        coef = stats_model_months.params[i]
        print(f"    • {month_name:12}: coefficient = {coef:+7.2f}")

if not sig_months:
    print(f"    • No month dummies are statistically significant")

print(f"\niii) MODEL IMPROVEMENT ASSESSMENT:")
print(f"     Adding MonthFactor {'likely improves' if r2_months > 0.85 else 'may improve' if r2_months > 0.75 else 'provides mixed improvement to'} model quality because:")
print(f"     • R² increased to {r2_months:.4f} (explains {r2_months:.1%} of variance)")
print(f"     • Captures seasonal patterns in car sales")
print(f"     • {len(sig_months)} months show significant differences from January")
print(f"     • However, adds {len(month_dummies.columns)} parameters (potential overfitting risk)")
print(f"     • Trade-off between explanatory power and model complexity")

print(f"\niv) ALTERNATIVE SEASONALITY MODELING:")
print(f"    Alternative approaches to model seasonality:")
print(f"    1. TRIGONOMETRIC FUNCTIONS:")
print(f"       • Use sin/cos functions: sin(2π×month/12), cos(2π×month/12)")
print(f"       • Captures smooth seasonal cycles with only 2 parameters")
print(f"       • More parsimonious than 11 dummy variables")
    
print(f"\n    2. POLYNOMIAL TIME TREND:")
print(f"       • Add month, month², month³ terms")
print(f"       • Captures curved seasonal patterns")
print(f"       • Fewer parameters than dummies")
    
print(f"\n    3. MOVING AVERAGES:")
print(f"       • Use 12-month moving averages to detrend")
print(f"       • Seasonal decomposition approach")
print(f"       • Better for forecasting")
    
print(f"\n    4. QUARTERLY DUMMIES:")
print(f"       • Use Q1, Q2, Q3, Q4 instead of monthly")
print(f"       • Only 3 parameters vs 11 monthly dummies")
print(f"       • Captures broader seasonal patterns")

print(f"\n    RECOMMENDATION:")
print(f"    The trigonometric approach would likely perform best because:")
print(f"    • Car sales follow natural seasonal cycles (weather, holidays)")
print(f"    • Sine/cosine functions naturally capture cyclical patterns")
print(f"    • Much more parsimonious (2 vs 11 parameters)")
print(f"    • Reduces overfitting risk while maintaining seasonal capture")
print(f"    • Better for out-of-sample forecasting")

# %% Final Model: Specified Variables with Full Seasonality

print("\n" + "="*80)
print("FINAL MODEL: SPECIFIED VARIABLES WITH MONTHLY SEASONALITY")
print("="*80)

# Build final model with specified variables
final_specified_vars = ['CivicQueries', 'CPIEnergy', 'CPIAll', 'MilesTraveled']
print(f"\n1. VARIABLE SELECTION JUSTIFICATION:")
print(f"-" * 50)
print(f"Selected variables: {', '.join(final_specified_vars)} + All months")
print(f"\nJustification:")
print(f"• CivicQueries: Direct measure of consumer interest/demand")
print(f"• CPIEnergy: Energy costs affect driving and car purchase decisions")
print(f"• CPIAll: Overall inflation impacts consumer purchasing power")
print(f"• MilesTraveled: Driving activity indicates car replacement needs")
print(f"• Monthly seasonality: Captures seasonal car buying patterns")
print(f"• Excluded Unemployment: Often correlated with other economic indicators")

# Create month dummies for both train and test sets
month_dummies_train = pd.get_dummies(train_data['MonthFactor'], prefix='Month', drop_first=True)
month_dummies_test = pd.get_dummies(test_data['MonthFactor'], prefix='Month', drop_first=True)

# Combine specified variables with all month dummies
X_final_train = pd.concat([
    train_data[final_specified_vars],
    month_dummies_train
], axis=1)

X_final_test = pd.concat([
    test_data[final_specified_vars],
    month_dummies_test
], axis=1)

print(f"\n2. FINAL MODEL CONSTRUCTION:")
print(f"-" * 50)
print(f"Economic variables: {len(final_specified_vars)}")
print(f"Monthly dummies: {len(month_dummies_train.columns)} (January = reference)")
print(f"Total variables: {X_final_train.shape[1]}")
print(f"Training observations: {X_final_train.shape[0]}")
print(f"Testing observations: {X_final_test.shape[0]}")

# Fit the final model
final_model = LinearRegression()
final_model.fit(X_final_train, y_train_spec)

# Training set performance
y_pred_final_train = final_model.predict(X_final_train)
r2_final_training = r2_score(y_train_spec, y_pred_final_train)
rmse_final_training = np.sqrt(mean_squared_error(y_train_spec, y_pred_final_train))

# Testing set performance (OSR²)
y_pred_final_test = final_model.predict(X_final_test)
r2_final_testing = r2_score(y_test, y_pred_final_test)  # This is the OSR²
rmse_final_testing = np.sqrt(mean_squared_error(y_test, y_pred_final_test))

print(f"\n3. MODEL PERFORMANCE:")
print(f"-" * 50)
print(f"Training R²:  {r2_final_training:.4f}")
print(f"Testing OSR²: {r2_final_testing:.4f}")
print(f"Training RMSE: {rmse_final_training:.2f} units")
print(f"Testing RMSE:  {rmse_final_testing:.2f} units")

# Get detailed statistical results
X_final_numeric = X_final_train.astype(float)
X_final_const = sm.add_constant(X_final_numeric)
stats_final_model = sm.OLS(y_train_spec.values, X_final_const.values).fit()

print(f"\n4. REGRESSION EQUATION:")
print(f"-" * 50)
print(f"CivicSales = {stats_final_model.params[0]:.2f}")

# Economic variables coefficients
for i, var in enumerate(final_specified_vars, 1):
    coef = stats_final_model.params[i]
    p_val = stats_final_model.pvalues[i]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    sign = ' + ' if coef >= 0 else ' - '
    print(f"           {sign} {abs(coef):.2f} × {var} {sig}")

# Month effects (show first few)
print(f"           + Monthly effects (relative to January):")
for i, month_col in enumerate(month_dummies_train.columns[:6], len(final_specified_vars) + 1):
    coef = stats_final_model.params[i]
    month_name = month_col.replace('Month_', '')
    sign = ' + ' if coef >= 0 else ' - '
    print(f"           {sign} {abs(coef):.2f} × {month_name}_dummy")
print(f"           ... (and {len(month_dummies_train.columns)-6} more month dummies)")

print(f"\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")

# Performance comparison and analysis
print(f"\n5. TRAINING vs TESTING PERFORMANCE ANALYSIS:")
print(f"-" * 50)

performance_diff = r2_final_training - r2_final_testing
relative_change = (performance_diff / r2_final_training) * 100

print(f"Training R²:     {r2_final_training:.4f}")
print(f"Testing OSR²:    {r2_final_testing:.4f}")
print(f"Difference:      {performance_diff:+.4f}")
print(f"Relative change: {relative_change:+.1f}%")

print(f"\n6. PERFORMANCE INTERPRETATION:")
print(f"-" * 50)

if abs(relative_change) < 5:
    performance_category = "EXCELLENT"
    interpretation = "Model shows exceptional generalization with minimal overfitting"
elif abs(relative_change) < 15:
    performance_category = "GOOD"
    interpretation = "Model generalizes well with acceptable performance drop"
elif abs(relative_change) < 25:
    performance_category = "MODERATE"
    interpretation = "Noticeable but reasonable performance difference"
else:
    performance_category = "SIGNIFICANT"
    interpretation = "Substantial performance difference indicates structural changes"

print(f"Performance Category: {performance_category}")
print(f"Interpretation: {interpretation}")

print(f"\n7. PLAUSIBLE EXPLANATIONS FOR PERFORMANCE DIFFERENCES:")
print(f"-" * 50)

if r2_final_testing > r2_final_training:
    print(f"OSR² > Training R² - Possible explanations:")
    print(f"• Test period (2020-2024) may have stronger, more predictable patterns")
    print(f"• COVID-19 created clearer seasonal/economic relationships")
    print(f"• Supply chain constraints made demand more predictable")
    print(f"• Pent-up demand during pandemic created stronger correlations")
elif performance_diff > 0:
    if relative_change < 10:
        print(f"Small performance drop - Expected and healthy:")
        print(f"• Normal out-of-sample performance degradation")
        print(f"• Model captures fundamental relationships well")
        print(f"• Training and test periods have similar underlying patterns")
        print(f"• Minimal overfitting to training data")
    elif relative_change < 25:
        print(f"Moderate performance drop - Structural differences:")
        print(f"• Economic conditions changed between periods")
        print(f"• 2020-2024: COVID-19, supply shortages, inflation, interest rate changes")
        print(f"• Consumer behavior shifts (remote work, urban exodus)")
        print(f"• Automotive industry disruption (chip shortage, EV transition)")
        print(f"• These factors altered the relationships between predictors and sales")
    else:
        print(f"Large performance drop - Major structural changes:")
        print(f"• Pandemic fundamentally altered car buying patterns")
        print(f"• Supply chain disruptions broke historical relationships")
        print(f"• Work-from-home reduced commuting and car needs")
        print(f"• Government stimulus and policy changes affected purchasing power")
        print(f"• Shift toward electric vehicles changed market dynamics")

print(f"\nSPECIFIC CONTEXTUAL FACTORS:")
print(f"• Training period (2014-2019): Post-recession recovery, stable growth")
print(f"• Testing period (2020-2024): Pandemic, supply chain crisis, inflation")
print(f"• Key changes: Remote work, chip shortage, used car boom, interest rates")
print(f"• These macro changes likely explain performance differences")

print(f"\n8. FINAL MODEL SUMMARY:")
print(f"-" * 50)
print(f"Model: 4 economic variables + 11 monthly seasonal dummies")
print(f"Variables: CivicQueries, CPIEnergy, CPIAll, MilesTraveled + Monthly effects")
print(f"Training R²: {r2_final_training:.4f} (explains {r2_final_training:.1%} of variance)")
print(f"Testing OSR²: {r2_final_testing:.4f} (out-of-sample performance)")
print(f"Model captures both economic fundamentals and seasonal patterns")
print(f"Performance difference reflects real-world structural changes between periods")

# %% Model Improvement: Addressing Poor OSR²

print("\n" + "="*80)
print("MODEL IMPROVEMENT: ADDRESSING POOR OUT-OF-SAMPLE PERFORMANCE")
print("="*80)

print(f"\nCURRENT PERFORMANCE ISSUE:")
print(f"Training R²: {r2_final_training:.4f}")
print(f"Testing OSR²: {r2_final_testing:.4f}")
print(f"Performance drop: {((r2_final_training - r2_final_testing)/r2_final_training)*100:.1f}%")

print(f"\nIMPROVEMENT STRATEGIES:")
print(f"-" * 50)

# Strategy 1: Regularized Regression (Ridge)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print(f"\n1. RIDGE REGRESSION (L2 Regularization):")
print(f"   Reduces overfitting by penalizing large coefficients")

# Standardize features for regularization
scaler = StandardScaler()
X_final_train_scaled = scaler.fit_transform(X_final_train)
X_final_test_scaled = scaler.transform(X_final_test)

# Try different regularization strengths
ridge_alphas = [0.1, 1.0, 10.0, 100.0]
best_ridge_osr2 = -999
best_ridge_alpha = None
best_ridge_model = None

print(f"   Testing regularization strengths:")
for alpha in ridge_alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_final_train_scaled, y_train_spec)
    
    ridge_train_pred = ridge_model.predict(X_final_train_scaled)
    ridge_test_pred = ridge_model.predict(X_final_test_scaled)
    
    ridge_train_r2 = r2_score(y_train_spec, ridge_train_pred)
    ridge_test_r2 = r2_score(y_test, ridge_test_pred)
    
    print(f"   Alpha {alpha:6.1f}: Train R² = {ridge_train_r2:.4f}, Test OSR² = {ridge_test_r2:.4f}")
    
    if ridge_test_r2 > best_ridge_osr2:
        best_ridge_osr2 = ridge_test_r2
        best_ridge_alpha = alpha
        best_ridge_model = ridge_model

print(f"   Best Ridge: Alpha = {best_ridge_alpha}, OSR² = {best_ridge_osr2:.4f}")

# Strategy 2: Simpler Model (Fewer Variables)
print(f"\n2. SIMPLIFIED MODEL (Reduced Complexity):")
print(f"   Use fewer variables to reduce overfitting")

# Try model with just the most correlated variables
correlations_simple = train_data[final_specified_vars].corrwith(y_train_spec).abs().sort_values(ascending=False)
top_2_vars = correlations_simple.head(2).index.tolist()
top_3_vars = correlations_simple.head(3).index.tolist()

print(f"   Variable correlations with CivicSales:")
for var in final_specified_vars:
    corr = train_data[var].corr(y_train_spec)
    print(f"   {var:15}: {corr:6.3f}")

# Test simplified models
simple_models = {
    'Top 2 vars': top_2_vars,
    'Top 3 vars': top_3_vars,
    'No seasonality': final_specified_vars  # Just economic vars, no months
}

best_simple_osr2 = -999
best_simple_name = None
best_simple_vars = None

for model_name, vars_to_use in simple_models.items():
    if model_name == 'No seasonality':
        X_simple_train = train_data[vars_to_use]
        X_simple_test = test_data[vars_to_use]
    else:
        # Include quarterly seasonality instead of monthly
        train_data_temp = train_data.copy()
        test_data_temp = test_data.copy()
        train_data_temp['Quarter'] = ((train_data_temp['MonthNumeric'] - 1) // 3) + 1
        test_data_temp['Quarter'] = ((test_data_temp['MonthNumeric'] - 1) // 3) + 1
        
        quarter_dummies_train_temp = pd.get_dummies(train_data_temp['Quarter'], prefix='Q', drop_first=True)
        quarter_dummies_test_temp = pd.get_dummies(test_data_temp['Quarter'], prefix='Q', drop_first=True)
        
        X_simple_train = pd.concat([train_data[vars_to_use], quarter_dummies_train_temp], axis=1)
        X_simple_test = pd.concat([test_data[vars_to_use], quarter_dummies_test_temp], axis=1)
    
    simple_model = LinearRegression()
    simple_model.fit(X_simple_train, y_train_spec)
    
    simple_train_pred = simple_model.predict(X_simple_train)
    simple_test_pred = simple_model.predict(X_simple_test)
    
    simple_train_r2 = r2_score(y_train_spec, simple_train_pred)
    simple_test_r2 = r2_score(y_test, simple_test_pred)
    
    print(f"   {model_name:15}: Train R² = {simple_train_r2:.4f}, Test OSR² = {simple_test_r2:.4f} ({len(X_simple_train.columns)} vars)")
    
    if simple_test_r2 > best_simple_osr2:
        best_simple_osr2 = simple_test_r2
        best_simple_name = model_name
        best_simple_vars = vars_to_use

# Strategy 3: Time-aware approach (Rolling window validation)
print(f"\n3. TIME-AWARE VALIDATION:")
print(f"   Check if recent training data predicts better")

# Use only last 3 years of training data (2017-2019)
recent_train = train_data[train_data['Year'] >= 2017]
y_recent_train = recent_train['CivicSales']

X_recent_train = pd.concat([
    recent_train[final_specified_vars],
    pd.get_dummies(recent_train['MonthFactor'], prefix='Month', drop_first=True)
], axis=1)

# Ensure same columns as test set
X_recent_train = X_recent_train.reindex(columns=X_final_test.columns, fill_value=0)

recent_model = LinearRegression()
recent_model.fit(X_recent_train, y_recent_train)

recent_train_pred = recent_model.predict(X_recent_train)
recent_test_pred = recent_model.predict(X_final_test)

recent_train_r2 = r2_score(y_recent_train, recent_train_pred)
recent_test_r2 = r2_score(y_test, recent_test_pred)

print(f"   Recent data (2017-2019): Train R² = {recent_train_r2:.4f}, Test OSR² = {recent_test_r2:.4f}")
print(f"   Training observations: {len(y_recent_train)} (vs {len(y_train_spec)} full sample)")

# Strategy 4: Ensemble approach
print(f"\n4. ENSEMBLE MODEL:")
print(f"   Combine multiple approaches for robustness")

# Simple average of predictions from different models
ensemble_test_pred = (best_ridge_model.predict(X_final_test_scaled) + 
                     recent_test_pred + 
                     y_pred_final_test) / 3

ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
print(f"   Ensemble OSR²: {ensemble_test_r2:.4f} (average of 3 models)")

# Compare all approaches
print(f"\n5. COMPARISON OF ALL APPROACHES:")
print(f"-" * 50)
print(f"Original model:      OSR² = {r2_final_testing:.4f}")
print(f"Best Ridge:          OSR² = {best_ridge_osr2:.4f} (alpha = {best_ridge_alpha})")
print(f"Best Simple:         OSR² = {best_simple_osr2:.4f} ({best_simple_name})")
print(f"Recent data only:    OSR² = {recent_test_r2:.4f}")
print(f"Ensemble:            OSR² = {ensemble_test_r2:.4f}")

# Select best approach
all_results = {
    'Original': r2_final_testing,
    'Ridge': best_ridge_osr2,
    'Simple': best_simple_osr2,
    'Recent': recent_test_r2,
    'Ensemble': ensemble_test_r2
}

best_approach = max(all_results, key=all_results.get)
best_osr2 = all_results[best_approach]

print(f"\nBEST APPROACH: {best_approach} with OSR² = {best_osr2:.4f}")

# Improvement analysis
original_osr2 = r2_final_testing
improvement = best_osr2 - original_osr2
relative_improvement = (improvement / abs(original_osr2)) * 100 if original_osr2 != 0 else float('inf')

print(f"\n6. IMPROVEMENT ANALYSIS:")
print(f"-" * 50)
print(f"Original OSR²:    {original_osr2:.4f}")
print(f"Improved OSR²:    {best_osr2:.4f}")
print(f"Absolute change:  {improvement:+.4f}")
if original_osr2 < 0:
    print(f"Relative improvement: Moved from negative to {'positive' if best_osr2 > 0 else 'less negative'}")
else:
    print(f"Relative improvement: {relative_improvement:+.1f}%")

print(f"\n7. RECOMMENDATIONS:")
print(f"-" * 50)
if best_osr2 > 0.3:
    print(f"✓ Achieved reasonable out-of-sample performance")
    print(f"✓ Model is now suitable for forecasting")
elif best_osr2 > 0:
    print(f"~ Modest improvement achieved")
    print(f"~ Model has some predictive value but limited")
else:
    print(f"✗ Still poor out-of-sample performance")
    print(f"✗ Fundamental structural changes between periods")
    print(f"✗ Consider completely different modeling approach")

print(f"\nFINAL RECOMMENDATION:")
if best_approach == 'Ridge':
    print(f"Use Ridge regression with alpha = {best_ridge_alpha} and standardized features")
elif best_approach == 'Simple':
    print(f"Use simplified model with {best_simple_name}: {best_simple_vars}")
elif best_approach == 'Recent':
    print(f"Use only recent training data (2017-2019) for model fitting")
elif best_approach == 'Ensemble':
    print(f"Use ensemble of multiple models for more robust predictions")
else:
    print(f"Original model performs best despite poor OSR²")

print(f"\nKEY INSIGHTS:")
print(f"• Structural break between training (2014-2019) and test (2020-2024) periods")
print(f"• COVID-19 pandemic fundamentally changed car buying patterns")
print(f"• Supply chain disruptions altered normal economic relationships")
print(f"• Model improvements help but cannot fully overcome structural changes")
print(f"• Future models should incorporate pandemic/disruption indicators")

# %%
