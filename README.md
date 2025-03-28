# 🌍 Global Livability Analysis Dashboard

## 📌 Overview
This project is a **data-driven dashboard** that ranks countries based on **livability metrics**. It combines **public sentiment analysis** with **economic and environmental indicators** to provide policymakers and investors with actionable insights.

🔹 **Sentiment Analysis:** Extracts **public opinions from Reddit** and categorizes sentiment for different aspects (Healthcare, Economy, Environment, etc.).
🔹 **World Bank Data Analysis:** Uses **historical data and Facebook Prophet** to **forecast livability scores** for each country.

Built using **Python, Streamlit, Plotly, and Machine Learning**.

---
## 🚀 Features
### 📊 1. Sentiment Analysis
- **Real-time Reddit data extraction**
- **NLP-based sentiment classification** (Strongly Negative → Strongly Positive)
- **Categorization of discussions into Healthcare, Economy, Environment, etc.**
- **Geographical mapping** of sentiment trends across countries
- **Interactive filters** for policymakers to explore specific concerns

### 🔮 2. Livability Score Forecasting (World Bank Data + Facebook Prophet)
- **Data gathered from the World Bank API**
- **Facebook Prophet time-series forecasting** (Predicts future livability rankings till 2027)
- **Comparison of historical vs. forecasted scores**
- **Key indicator analysis:** GDP per capita, Life Expectancy, Literacy Rate, CO2 Emissions, etc.
- **Factor correlation analysis:** Identifies which indicators influence livability the most

### 📌 3. Dashboard Functionalities
- **Interactive visualizations** (Plotly charts, geo-maps, bar charts)
- **Policy recommendations based on sentiment trends**
- **Country-wise comparison & ranking system**
- **Modern UI with custom CSS for an enhanced experience**

---
## ⚡ Tech Stack
- **Frontend:** Streamlit, Plotly, HTML/CSS
- **Backend:** Python (Flask for API handling)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Facebook Prophet (for forecasting), NLP (Sentiment Analysis)

---
## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/livability-dashboard.git
cd livability-dashboard
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Dashboard
```bash
streamlit run dashboard.py
```

---


---
## 📌 Future Enhancements
✅ Integrate **real-time sentiment monitoring** 📊
✅ Improve **forecasting models with advanced ML techniques** 📡
✅ Add **more country-specific factors** for deeper analysis 🌍

---
## 🤝 Contributing
Feel free to open an **issue** or submit a **pull request** to enhance the dashboard!

---
## 📜 License
This project is licensed under the **MIT License**.
