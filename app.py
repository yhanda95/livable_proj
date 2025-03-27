import streamlit as st
import base64
import os
import pandas as pd
from datetime import datetime
from predict_page import load_data
import dashboard as dashboard
import google.generativeai as genai

genai.configure(api_key="AIzaSyBN4g03--SHgVne8N3cnuhxKF5r25qpYac")  # 🔹 Replace with your actual key

def query_gemini(question):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  # Use Gemini-Pro model
        response = model.generate_content(question)
        return response.text  # Extract AI-generated response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Livability Analysis App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(file_path):
    try:
        bin_str = get_base64_of_bin_file(file_path)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")

def load_custom_css():
    # Load Google Fonts
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Montserrat:wght@400;500;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    image_path = os.path.join(os.path.dirname(__file__), 'static', 'background.jpg')
    set_background(image_path)
    
    st.markdown("""
    <style>
    /* Global styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Main container */
    .main-container {
        text-align: center;
        padding: 3rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Title styling */
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #1a3c5e;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.25rem;
        font-weight: 300;
        color: #34495e;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    /* Button styling */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .custom-button {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        font-weight: 600;
        border: none;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        text-align: center;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .custom-button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
    }
    
    .custom-button:active {
        transform: translateY(1px);
    }
    
    /* Feature cards */
    .feature-section {
        display: flex;
        gap: 1.5rem;
        margin-top: 2.5rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        flex: 1;
        transition: transform 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #3498db;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        font-size: 0.95rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    
    /* About section */
    .about-section {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .about-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Stats counter section */
    .stats-section {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background:linear-gradient(135deg, #3498db, #1a5276);
        border-radius: 12px;
        color: white;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ECF0F1 ;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Footer */
    .footer {
        background: rgba(44, 62, 80, 0.9);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .footer-link {
        color: #3498db;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: #2980b9;
        text-decoration: underline;
    }
    
    /* Notifications */
    .notification {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        animation: slideIn 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Dashboard container */
    .dashboard-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar customization */
    .css-1d391kg, .css-12oz5g7 {
        background-color: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-container {
            padding: 1.5rem;
        }
        
        .title {
            font-size: 2rem;
        }
        
        .feature-section {
            flex-direction: column;
        }
        
        .stats-section {
            flex-wrap: wrap;
        }
        
        .stat-item {
            flex: 1 0 40%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def display_notification(message, type="info"):
    color = {
        "info": "#3498db",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "error": "#e74c3c"
    }
    
    st.markdown(f"""
    <div class="notification" style="background-color: {color[type]}20; border-left: 4px solid {color[type]};">
        <div>
            <strong style="color: {color[type]};">{"✅ " if type == "success" else "ℹ️ " if type == "info" else "⚠️ " if type == "warning" else "❌ "}</strong>
            <span>{message}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def animated_stats():
    st.markdown("""
    <div class="stats-section">
        <div class="stat-item">
            <div class="stat-number">200+</div>
            <div class="stat-label">Countries Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">15+</div>
            <div class="stat-label">Livability Factors</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">5M+</div>
            <div class="stat-label">Data Points</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">98%</div>
            <div class="stat-label">Prediction Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def feature_cards():
    st.markdown("""
    <div class="feature-section">
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Deep Analysis</div>
            <div class="feature-description">
                Explore comprehensive visual insights with our interactive dashboard. Compare different metrics across countries and regions.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Predictive Models</div>
            <div class="feature-description">
                Use advanced machine learning algorithms to predict future trends in health, education, and economic factors.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Sentiment Analysis</div>
            <div class="feature-description">
                Monitor real-time public sentiment and discover how people feel about livability factors around the world.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_footer():
    current_year = datetime.now().year
    st.markdown(f"""
    <div class="footer">
        <p>© {current_year} Livability Analysis Project. All rights reserved.</p>
        <div class="footer-links">
            <a href="#" class="footer-link">Privacy Policy</a>
            <a href="#" class="footer-link">Terms of Service</a>
            <a href="#" class="footer-link">Contact Us</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def landing_page():
    st.markdown("""
    <div class="main-container">
        <h1 class="title">🌍 Global Livability Analysis</h1>
        <p class="subtitle">Gain valuable insights into global livability factors through interactive visualizations, 
        predictive analytics, and real-time sentiment analysis from social media data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Streamlit button instead of HTML button
    if st.button("Learn More ↓", key="learn_more"):
        st.session_state['show_more_info'] = True
    
    # Display the about section if button is clicked
    if 'show_more_info' in st.session_state and st.session_state['show_more_info']:
        st.markdown("""
        <div class="about-section">
            <h2 class="about-title">🌟 What We Do</h2>
            <p>
            The Global Livability Analysis project aims to provide comprehensive tools for understanding, comparing, and predicting quality of life metrics across different countries and regions...
            </p>
            <strong>🔍 Ask AI:</strong> An AI-powered chatbot integrated into the web app to provide instant assistance and personalized user interactions.
            </p>
            <p>
            <strong>📈 Predictive Analytics Tool:</strong> Leverage machine learning to predict future trends in health, education, and economic factors. Make data-driven decisions with our advanced predictive models.
            </p>
            <p>
            <strong>📊 Global Sentiment Explorer:</strong> Monitor real-time public sentiment from social media and discover how people feel about livability factors around the world.
            </p>
            
        </div>
        """, unsafe_allow_html=True)
    
    animated_stats()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔍 Ask AI", key="ai_button", use_container_width=True):
            st.session_state['page'] = "AI"
    
    with col2:
        if st.button("📈 Predictive Analytics Tool", key="predict_button", use_container_width=True):
            st.session_state['page'] = "Predict"
    
    with col3:
        if st.button("📊 Global Sentiment Explorer", key="sentiment_dashboard_button", use_container_width=True):
            st.session_state['page'] = "SentimentDashboard"
    
    feature_cards()
    
    
    create_footer()

def analyze_page():
    st.header("Ask AI about the data or general topics!")
    user_input = st.text_area("Enter your question:")
    if st.button("Ask AI"):
        if user_input:
            response = query_gemini(user_input)
            st.write(response)
        else:
            st.warning("Please enter a question.")
   
    
    if st.button("⬅️ Go Back to Landing Page", key="go_back_analyze"):
        st.session_state['page'] = "Landing"

def predict_page():
    load_data()
    
    if st.button("⬅️ Go Back to Landing Page", key="go_back_predict"):
        st.session_state['page'] = "Landing"

def sentiment_dashboard_page():
    dashboard.run_dashboard()  # Call the dashboard function
    
    if st.button("⬅️ Go Back to Landing Page", key="go_back_sentiment"):
        st.session_state['page'] = "Landing"

def main():
    load_custom_css()
    
    if 'page' not in st.session_state:
        st.session_state['page'] = "Landing"
    
    if st.session_state['page'] == "Landing":
        landing_page()
    elif st.session_state['page'] == "AI":
        analyze_page()
    elif st.session_state['page'] == "Predict":
        predict_page()
    elif st.session_state['page'] == "SentimentDashboard":
        sentiment_dashboard_page()

if __name__ == "__main__":
    main()
            