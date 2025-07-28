import pandas as pd
import mysql.connector

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="navya",        
    password="your_password",
    database="livability"
)
cursor = conn.cursor()

# Dataset configuration: table name -> file path
datasets = {
    "historical_scores": "dashboard_data/historical_scores.csv",
    "forecasts": "dashboard_data/forecasts.csv",
    "country_summary": "dashboard_data/country_summary.csv",
    "top_countries": "dashboard_data/top_countries.csv",
    "bottom_countries": "dashboard_data/bottom_countries.csv",
    "factor_correlations": "dashboard_data/factor_correlations.csv"
}

# Column renaming for historical_scores only
def clean_historical_columns(df):
    df.rename(columns={
        "Forest area (% of land area)": "Forest_area_percent",
        "GDP per capita (current US$)": "GDP_per_capita",
        "Life expectancy at birth (years)": "Life_expectancy",
        "Literacy rate (% of people ages 15 and above)": "Literacy_rate",
        "Unemployment rate (% of total labor force)": "Unemployment_rate",
        "Electricity access (% of population)": "Electricity_access",
        "Renewable energy consumption (% of total energy use)": "Renewable_energy_percent",
        "Inflation (annual %)": "Inflation_percent"
    }, inplace=True)
    return df

# Universal insert function
def insert_dataframe(df, table_name):
    # Handle reserved keywords (like `Change`)
    columns = ", ".join([f"`{col}`" for col in df.columns])
    placeholders = ", ".join(["%s"] * len(df.columns))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    for row in df.itertuples(index=False, name=None):
        cursor.execute(query, row)
    conn.commit()
    print(f"✅ {len(df)} rows inserted into `{table_name}`")

# Process each dataset
for table, file_path in datasets.items():
    df = pd.read_csv(file_path)

    if table == "historical_scores":
        df = clean_historical_columns(df)

    insert_dataframe(df, table)

# Clean up
cursor.close()
conn.close()
print("🎯 All tables populated successfully.")

