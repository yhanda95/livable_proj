import requests
import pandas as pd

# List of all 195 countries (ISO Alpha-3 codes)
country_codes = ["AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "AUS", "AUT", "AZE", "BHS", "BHR", "BGD", "BRB", 
                 "BLR", "BEL", "BLZ", "BEN", "BTN", "BOL", "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM", 
                 "CMR", "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD", "CRI", "CIV", "HRV", "CUB", "CYP", 
                 "CZE", "DNK", "DJI", "DMA", "DOM", "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN", 
                 "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", "GNB", "GUY", "HTI", "HND", "HUN", 
                 "ISL", "IND", "IDN", "IRN", "IRQ", "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT", 
                 "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", "MDG", "MWI", "MYS", "MDV", "MLI", 
                 "MLT", "MHL", "MRT", "MUS", "MEX", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NRU", "NPL", 
                 "NLD", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN", "PAK", "PLW", "PAN", "PNG", "PRY", "PER", 
                 "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "RWA", "WSM", "SMR", "STP", "SAU", "SEN", "SRB", "SYC", "SLE", 
                 "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF", "KOR", "SSD", "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", 
                 "TWN", "TJK", "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV", "UGA", "UKR", "ARE", 
                 "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM", "YEM", "ZMB", "ZWE"]

# Indicators of interest
indicators = {
    "EN.ATM.CO2E.PC": "CO2 emissions (metric tons per capita)",
    "SH.H2O.SAFE.ZS": "Access to safe drinking water (% of population)",
    "AG.LND.FRST.ZS": "Forest area (% of land area)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "SP.POP.TOTL": "Population total",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total)",
    "SP.DYN.CBRT.IN": "Birth rate (per 1,000 people)",
    "SP.DYN.CDRT.IN": "Death rate (per 1,000 people)",
    "SP.DYN.LE00.IN": "Life expectancy at birth (years)",
    "SE.ADT.LITR.ZS": "Literacy rate (% of people ages 15 and above)",
    "SL.UEM.TOTL.ZS": "Unemployment rate (% of total labor force)",
    "EG.ELC.ACCS.ZS": "Electricity access (% of population)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total energy use)",
    "FP.CPI.TOTL.ZG": "Inflation (annual %)"
}

# API endpoint template
base_url = "http://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=10000"

# Years to extract data for
years = list(range(2000, 2023))  # From 2000 to 2022

# Initialize list to store data
all_data = []

# Fetch data for each country and indicator
for country in country_codes:
    for indicator, indicator_name in indicators.items():
        print(f"Fetching {indicator_name} for {country}...")
        response = requests.get(base_url.format(country, indicator))
        
        if response.status_code == 200:
            data = response.json()
            
            # Ensure the response has valid data
            if len(data) > 1 and isinstance(data[1], list):
                records = data[1]  # Extract records
                
                for record in records:
                    year = record.get("date")
                    value = record.get("value")

                    if year and int(year) in years:
                        all_data.append({
                            "Country": country,
                            "Year": year,
                            "Indicator": indicator_name,
                            "Value": value
                        })
            else:
                print(f"No data found for {indicator_name} in {country}.")
        else:
            print(f"Failed to fetch {indicator_name} for {country} (Status Code: {response.status_code})")

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Pivot the DataFrame for better readability
df_pivot = df.pivot_table(index=["Country", "Year"], columns="Indicator", values="Value").reset_index()

# Save to CSV
df_pivot.to_csv("world_bank_195_countries.csv", index=False)

print("Data extraction complete! Saved as 'world_bank_195_countries.csv'.")
