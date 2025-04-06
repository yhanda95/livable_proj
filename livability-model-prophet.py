import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('dashboard_data', exist_ok=True)
os.makedirs('model_evaluation', exist_ok=True)

# Load the dataset
df = pd.read_csv("updated_world_bank_195_countries_complete.csv")

# Basic data exploration
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Number of countries: {df['Country'].nunique()}")
print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

# Check missing values before processing
missing_percent = df.isnull().mean() * 100
print("\nMissing values percentage before processing:")
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# Define factors contributing to livability with their direction (positive/negative impact)
livability_factors = {
    'Access to safe drinking water (% of population)': 1,  # higher is better
    'Forest area (% of land area)': 1,  # higher is better
    'GDP per capita (current US$)': 1,  # higher is better
    'Life expectancy at birth (years)': 1,  # higher is better
    'Literacy rate (% of people ages 15 and above)': 1,  # higher is better
    'Unemployment rate (% of total labor force)': -1,  # lower is better
    'Electricity access (% of population)': 1,  # higher is better
    'Renewable energy consumption (% of total energy use)': 1,  # higher is better
    'CO2 emissions (metric tons per capita)': -1,  # lower is better
    'Inflation (annual %)': -1  # lower is better
}

# Check if all factors exist in the dataset
missing_factors = [factor for factor in livability_factors.keys() if factor not in df.columns]
if missing_factors:
    print(f"Warning: The following factors are missing from the dataset: {missing_factors}")
    # Remove missing factors from our dictionary
    for factor in missing_factors:
        del livability_factors[factor]

print(f"\nUsing the following factors for livability score calculation:")
for factor, direction in livability_factors.items():
    print(f"- {factor} ({'positive' if direction == 1 else 'negative'} impact)")

# Create a copy to avoid modifying the original DataFrame
df_processed = df.copy()

# Process missing values separately for each country and factor
print("\nProcessing missing values by country and factor...")

# Group data by country
country_groups = df_processed.groupby('Country')

# For each factor, impute missing values within each country
for factor in livability_factors.keys():
    if factor in df_processed.columns:
        # For each country, impute missing values using either forward fill, backward fill, or mean
        for country, group in country_groups:
            country_mask = df_processed['Country'] == country
            
            # Check if there are any missing values for this factor in this country
            if df_processed.loc[country_mask, factor].isnull().any():
                # Try forward fill first (use previous years' values)
                df_processed.loc[country_mask, factor] = df_processed.loc[country_mask, factor].ffill()
                
                # Then try backward fill (use future years' values for any remaining NaNs)
                df_processed.loc[country_mask, factor] = df_processed.loc[country_mask, factor].bfill()
                
                # If still have NaNs, use the global mean for that factor
                if df_processed.loc[country_mask, factor].isnull().any():
                    global_mean = df_processed[factor].mean()
                    df_processed.loc[country_mask, factor] = df_processed.loc[country_mask, factor].fillna(global_mean)

# Check remaining missing values after imputation
missing_after = df_processed.isnull().mean() * 100
print("\nMissing values percentage after imputation:")
print(missing_after[missing_after > 0].sort_values(ascending=False))

# Normalize the data for each indicator (min-max scaling)
print("\nNormalizing factors...")
scaler = MinMaxScaler()
df_scaled = df_processed.copy()

for factor in livability_factors.keys():
    if factor in df_processed.columns:
        # Scale the factor
        df_scaled[factor] = scaler.fit_transform(df_processed[factor].values.reshape(-1, 1))
        
        # Invert negative factors (so higher is always better)
        if livability_factors[factor] == -1:
            df_scaled[factor] = 1 - df_scaled[factor]

# Calculate livability score (average of normalized indicators)
livability_columns = [col for col in livability_factors.keys() if col in df_scaled.columns]
df_scaled['Livability_Score'] = df_scaled[livability_columns].mean(axis=1)

# Add the score to the processed dataframe
df_processed['Livability_Score'] = df_scaled['Livability_Score']

# Time Series Forecasting using Facebook Prophet with train-test split and evaluation
print("\nForecasting livability scores for countries using Prophet with evaluation...")

def evaluate_prophet_model(train_df, test_df):
    """Evaluate Prophet model performance on test set"""
    try:
        # Prepare training data
        train_data = train_df.reset_index()[['Year', 'Livability_Score']].rename(
            columns={'Year': 'ds', 'Livability_Score': 'y'})
        train_data['ds'] = pd.to_datetime(train_data['ds'], format='%Y')
        
        # Prepare test data
        test_data = test_df.reset_index()[['Year', 'Livability_Score']].rename(
            columns={'Year': 'ds', 'Livability_Score': 'y'})
        test_data['ds'] = pd.to_datetime(test_data['ds'], format='%Y')
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=False,
            growth='linear',
            interval_width=0.95,
            changepoint_prior_scale=0.1
        )
        model.fit(train_data)
        
        # Make predictions on test set
        forecast = model.predict(test_data[['ds']])
        
        # Calculate evaluation metrics
        y_true = test_data['y'].values
        y_pred = forecast['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
       
        return {
            'mae': mae,
            'rmse': rmse,
            'y_true': y_true,
            'y_pred': y_pred,
            'test_years': test_data['ds'].dt.year.values
        }
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return None

def forecast_prophet(country_df, periods=5, min_data_points=5, evaluate=False, target_years=None):
    """Forecast with optional evaluation and specific target years"""
    if len(country_df) < min_data_points:
        return None, None
    
    # Split data into train and test sets if evaluating
    eval_results = None
    if evaluate and len(country_df) >= 8:  # Need enough data for meaningful evaluation
        train_size = int(len(country_df) * 0.7)
        train_df = country_df.iloc[:train_size]
        test_df = country_df.iloc[train_size:]
        
        eval_results = evaluate_prophet_model(train_df, test_df)
        country_df = train_df  # Use only training data for final forecasting
    
    prophet_df = country_df.reset_index()[['Year', 'Livability_Score']].rename(
        columns={'Year': 'ds', 'Livability_Score': 'y'})
    
    if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
    
    model = Prophet(
        yearly_seasonality=False,
        growth='linear',
        interval_width=0.95,
        changepoint_prior_scale=0.1
    )
    
    try:
        model.fit(prophet_df)
        
        # Use fixed target years if provided, otherwise calculate based on last year
        if target_years is not None:
            future_years = pd.DataFrame({'ds': [pd.Timestamp(year=year, month=1, day=1) for year in target_years]})
        else:
            # Get the last year in the dataset
            last_year = prophet_df['ds'].dt.year.max()
            # Generate future years starting from the year after the last year
            forecast_years = pd.date_range(start=f'{last_year+1}-01-01', periods=periods, freq='Y')
            future_years = pd.DataFrame({'ds': forecast_years})
        
        forecast = model.predict(future_years)
        
        forecast['yhat'] = np.clip(forecast['yhat'], 0, 1)
        forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, 1)
        forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, 1)
        
        return forecast, eval_results
    except Exception as e:
        print(f"  Prophet modeling error: {str(e)}")
        return None, None

# Define the specific target years we want to forecast (2023-2027)
target_forecast_years = list(range(2023, 2028))
print(f"\nForcing forecasts to use specific years: {target_forecast_years}")

all_forecasts = {}
failed_countries = []
evaluation_results = []
min_data_points = 4

for country in df_processed['Country'].unique():
    country_data = df_processed[df_processed['Country'] == country].copy()
    country_data_indexed = country_data.set_index('Year').sort_index()
    
    years = sorted(country_data_indexed.index.unique())
    if len(years) >= min_data_points:
        has_large_gaps = False
        for i in range(1, len(years)):
            if years[i] - years[i-1] > 3:
                has_large_gaps = True
                break
        
        if not has_large_gaps:
            try:
                # Only evaluate for countries with sufficient data
                evaluate = len(years) >= 8
                forecast, eval_result = forecast_prophet(
                    country_data_indexed, 
                    periods=5, 
                    min_data_points=min_data_points,
                    evaluate=evaluate,
                    target_years=target_forecast_years  # Use our fixed target years
                )
                
                if eval_result:
                    evaluation_results.append({
                        'country': country,
                        'mae': eval_result['mae'],
                        'rmse': eval_result['rmse'],
                        'test_years': eval_result['test_years'].tolist(),
                        'y_true': eval_result['y_true'].tolist(),
                        'y_pred': eval_result['y_pred'].tolist()
                    })
                
                if forecast is not None and not forecast.empty:
                    # Store the forecasted years properly
                    all_forecasts[country] = {
                        'historical': country_data_indexed['Livability_Score'].to_dict(),
                        'forecast': forecast['yhat'].values,
                        'forecast_lower': forecast['yhat_lower'].values,
                        'forecast_upper': forecast['yhat_upper'].values,
                        'latest_score': country_data_indexed['Livability_Score'].iloc[-1],
                        'years': forecast['ds'].dt.year.values  # This will now be 2023-2027
                    }
                    print(f"  Forecasted livability scores for {country} (2023-2027): {forecast['yhat'].values.round(3)}")
                else:
                    failed_countries.append(country)
            except Exception as e:
                print(f"  Could not forecast for {country}. Error: {str(e)}")
                failed_countries.append(country)
        else:
            failed_countries.append(country)
    else:
        failed_countries.append(country)

# Create comprehensive forecast dataframe
forecast_data = []
for country, data in all_forecasts.items():
    for i, year in enumerate(data['years']):
        forecast_data.append({
            'Country': country,
            'Year': year,
            'Livability_Score': data['forecast'][i],
            'Lower_Bound': data['forecast_lower'][i],
            'Upper_Bound': data['forecast_upper'][i],
            'Latest_Score': data['latest_score']
        })

forecast_df = pd.DataFrame(forecast_data)

# Generate country report
country_report = []
for country in all_forecasts.keys():
    latest_year = max(all_forecasts[country]['historical'].keys())
    latest_score = all_forecasts[country]['historical'][latest_year]
    forecast_2027 = all_forecasts[country]['forecast'][-1]  # Using 2027 as last year
    change = forecast_2027 - latest_score
    
    country_report.append({
        'Country': country,
        'Latest_Score': latest_score,
        'Latest_Year': latest_year,
        'Forecast_2027': forecast_2027,
        'Change': change,
        'Percent_Change': (change / latest_score * 100).round(2) if latest_score > 0 else 0
    })

country_report_df = pd.DataFrame(country_report).sort_values('Forecast_2027', ascending=False)

# Calculate factor correlations
correlation = df_processed[list(livability_factors.keys()) + ['Livability_Score']].corr()['Livability_Score'].sort_values(ascending=False)

# Save evaluation results
if evaluation_results:
    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv('model_evaluation/prophet_evaluation_results.csv', index=False)
    
    # Calculate average metrics
    avg_metrics = {
        'MAE': eval_df['mae'].mean(),
        'RMSE': eval_df['rmse'].mean(),
    }
    
    # Plot evaluation metrics
    plt.figure(figsize=(10, 6))
    metrics = ['MAE', 'RMSE']
    values = avg_metrics['MAE'], avg_metrics['RMSE']
    
    plt.bar(metrics, values, color=['#4e79a7', '#f28e2b'])
    plt.title('Average Prophet Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig('model_evaluation/average_metrics.png')
    plt.close()

# Save all dashboard data
def save_dashboard_data():
    # Historical data with all factors
    df_processed[['Country', 'Year'] + list(livability_factors.keys()) + ['Livability_Score']]\
        .to_csv('dashboard_data/historical_scores.csv', index=False)
    
    # Forecast data
    forecast_df.to_csv('dashboard_data/forecasts.csv', index=False)
    
    # Country summary
    country_report_df.to_csv('dashboard_data/country_summary.csv', index=False)
    
    # Factor correlations
    correlation.drop('Livability_Score').reset_index()\
        .rename(columns={'index': 'Factor', 'Livability_Score': 'Correlation'})\
        .to_csv('dashboard_data/factor_correlations.csv', index=False)
    
    # Top and bottom countries
    top_countries = country_report_df.head(10)[['Country', 'Latest_Score', 'Forecast_2027', 'Percent_Change']]
    bottom_countries = country_report_df.tail(10)[['Country', 'Latest_Score', 'Forecast_2027', 'Percent_Change']]
    
    top_countries.to_csv('dashboard_data/top_countries.csv', index=False)
    bottom_countries.to_csv('dashboard_data/bottom_countries.csv', index=False)

save_dashboard_data()

# Print summary statistics
print("\nAnalysis Summary:")
print(f"- Total countries analyzed: {len(all_forecasts)}")
print(f"- Countries with insufficient data or errors: {len(failed_countries)}")
print(f"- Forecast period: 2023 to 2027")

if evaluation_results:
    print("\nModel Evaluation Summary:")
    print(f"- Average MAE: {avg_metrics['MAE']:.4f}")
    print(f"- Average RMSE: {avg_metrics['RMSE']:.4f}")

print("\nTop 10 countries by forecasted 2027 livability score:")
print(country_report_df.head(10)[['Country', 'Latest_Score', 'Forecast_2027', 'Percent_Change']])

print("\nBottom 10 countries by forecasted 2027 livability score:")
print(country_report_df.tail(10)[['Country', 'Latest_Score', 'Forecast_2027', 'Percent_Change']])

print("\nFactor importance (correlation with livability score):")
print(correlation.drop('Livability_Score'))

for result in evaluation_results:
    print(f"Country: {result['country']}")
    print(f"Train Years: {set(range(2000, 2023)) - set(result['test_years'])}")
    print(f"Test Years: {result['test_years']}\n")


print("\nAnalysis complete. All files saved:")
print("- Dashboard data in dashboard_data/ directory")