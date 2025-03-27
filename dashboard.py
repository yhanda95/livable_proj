# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

def run_dashboard():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main {background-color: #f5f5f5;}
        .sidebar .sidebar-content {background-color: #003366;max-width: 150px;}
        h1 {color: #003366;}
        h2 {color: #2e5984;}
        h3 {color: #4a77a8;}
        .metric-label {font-size: 14px; color: #666;}
        .metric-value {font-size: 24px; color: #003366; font-weight: bold;}
        .range-info {font-size: 12px; color: #666; font-style: italic;}
        
        /* Card styling */
.card {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    background-color: white;
    margin-bottom: 20px;
    height: 100%;  /* Add this */
    min-height: 180px;  /* Add minimum height */
    display: flex;  /* Add flexbox */
    flex-direction: column;  /* Stack content vertically */
    justify-content: space-between;  /* Distribute space */
}

/* Add container styling for metric columns */
[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
}

/* Ensure content fills card space */
.card-header {
    flex-shrink: 0;
}
.metric-value {
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
        
        /* Table styling */
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 14px;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #003366;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th,
        .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #003366;
        }
        
        /* Badge styling */
        .badge {
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-red {
            background-color: #e74c3c;
            color: white;
        }
        .badge-orange {
            background-color: #f39c12;
            color: white;
        }
        .badge-yellow {
            background-color: #f1c40f;
            color: white;
        }
        .badge-green {
            background-color: #2ecc71;
            color: white;
        }
        .badge-blue {
            background-color: #3498db;
            color: white;
        }
        .badge-gray {
            background-color: #95a5a6;
            color: white;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            background-color: #f0f0f0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #003366;
            color: white;
        }
        
        /* Button styling */
        .download-btn {
            background-color: #003366;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
        }
        
        /* Insight cards */
        .insight-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            height: 100%;
        }
        .insight-card-title {
            font-size: 16px;
            font-weight: bold;
            color: #003366;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .insight-card-content {
            font-size: 14px;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-red {
            background-color: #e74c3c;
        }
        .status-orange {
            background-color: #f39c12;
        }
        .status-yellow {
            background-color: #f1c40f;
        }
        .status-green {
            background-color: #2ecc71;
        }
        .status-blue {
            background-color: #3498db;
        }
        .status-gray {
            background-color: #95a5a6;
        }
        </style>
        """, unsafe_allow_html=True)

    # SECTION: Data Loading
    @st.cache_data
    def load_data():
        data_dir = "analysis_results"
        data_files = [f for f in os.listdir(data_dir) if f.startswith("analyzed_data")]
        if not data_files:
            st.error("No analysis data found. Please run data collection and analysis first.")
            return None
        
        latest_file = sorted(data_files)[-1]
        df = pd.read_csv(os.path.join(data_dir, latest_file))
        
        # Load country coordinates
        world_data = pd.read_csv(r"world_data_with_scores.csv")
        return df.merge(world_data[['Country', 'Latitude', 'Longitude']], 
                       left_on='country', right_on='Country', how='left')

    # Helper functions for badges
    def get_sentiment_badge(sentiment_score):
        if sentiment_score < -0.3:
            return '<span class="badge badge-red">Strongly Negative</span>'
        elif sentiment_score < -0.1:
            return '<span class="badge badge-orange">Moderately Negative</span>'
        elif sentiment_score < 0.1:
            return '<span class="badge badge-gray">Neutral</span>'
        elif sentiment_score < 0.3:
            return '<span class="badge badge-yellow">Moderately Positive</span>'
        else:
            return '<span class="badge badge-green">Strongly Positive</span>'

    def get_priority_badge(priority):
        if priority == "High":
            return '<span class="badge badge-red">High Priority</span>'
        elif priority == "Medium":
            return '<span class="badge badge-orange">Medium Priority</span>'
        else:
            return '<span class="badge badge-blue">Low Priority</span>'

    def get_volatility_badge(volatility):
        if volatility >= 0.25:
            return '<span class="badge badge-red">High Volatility</span>'
        elif volatility >= 0.1:
            return '<span class="badge badge-orange">Medium Volatility</span>'
        else:
            return '<span class="badge badge-green">Low Volatility</span>'

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # SECTION: Filters
    with st.sidebar:
        st.header("Filters")
        filter_tabs = st.tabs(["Geographic", "Content", "Timeframe"])
        
        with filter_tabs[0]:
            selected_countries = st.multiselect(
                "Select Countries",
                options=df['country'].unique(),
                default=df['country'].unique()[:2]
            )
        
        with filter_tabs[1]:
            selected_categories = st.multiselect(
                "Select Categories",
                options=df['category'].unique(),
                default=['Health', 'Economy', 'Environment', 'Social']
            )
            
            selected_emotions = []
            if 'emotion' in df.columns:
                selected_emotions = st.multiselect(
                    "Filter by Emotions",
                    options=df['emotion'].dropna().unique(),
                    default=[]
                )
        
        with filter_tabs[2]:
            st.info("Timeframe filters will be enabled when time series data is available")
        
        if st.button("Apply Filters", type="primary"):
            st.success("Filters applied!")
        if st.button("Reset Filters", type="secondary"):
            st.info("Filters reset to default")

    # Apply filters
    filtered_df = df[df['country'].isin(selected_countries)]
    filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    if selected_emotions and 'emotion' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['emotion'].isin(selected_emotions)]

    # SECTION: Dashboard Header
    st.title("🌍 Public Sentiment Analysis Dashboard")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin: 0;">Actionable Insights for Policy Makers</h3>
            <p style="margin: 5px 0 0 0; font-style: italic;">Last updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

    # SECTION: Main Dashboard Content
    main_tabs = st.tabs(["Overview", "Sentiment Analysis", "Trend Insights", "Actionable Insights", "Documentation"])
    
    # TAB 1: Overview
    with main_tabs[0]:
        st.header("Key Metrics")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.markdown("""
            <div class="card">
                <div class="card-header">Total Mentions</div>
                <div class="metric-value">{:,}</div>
                <div class="range-info">({:.1f}% of total dataset)</div>
            </div>
            """.format(
                len(filtered_df),
                (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
            ), unsafe_allow_html=True)
        
        with metric_cols[1]:
            avg_sentiment = filtered_df['sentiment_score'].mean()
            sentiment_color = "#2ecc71" if avg_sentiment > 0.1 else "#e74c3c" if avg_sentiment < -0.1 else "#95a5a6"
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Average Sentiment</div>
                <div class="metric-value" style="color: {sentiment_color}">{avg_sentiment:.2f}</div>
                <div class="range-info">{get_sentiment_badge(avg_sentiment)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            if 'emotion' in filtered_df.columns and not filtered_df['emotion'].isnull().all():
                top_emotion = filtered_df['emotion'].value_counts().index[0]
                emotion_pct = (filtered_df['emotion'].value_counts().iloc[0] / len(filtered_df)) * 100
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Dominant Emotion</div>
                    <div class="metric-value">{top_emotion}</div>
                    <div class="range-info">Present in {emotion_pct:.1f}% of mentions</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card">
                    <div class="card-header">Dominant Emotion</div>
                    <div class="metric-value">N/A</div>
                    <div class="range-info">No emotion data available</div>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            if not filtered_df.empty and 'category' in filtered_df.columns:
                top_category = filtered_df['category'].value_counts().index[0]
                category_pct = (filtered_df['category'].value_counts().iloc[0] / len(filtered_df)) * 100
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Most Discussed Category</div>
                    <div class="metric-value">{top_category}</div>
                    <div class="range-info">Comprises {category_pct:.1f}% of discussions</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card">
                    <div class="card-header">Most Discussed Category</div>
                    <div class="metric-value">N/A</div>
                    <div class="range-info">No category data available</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Interpretation Guide
        with st.expander("📊 Metrics Interpretation Guide for Policy Makers", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <div class="card-header">Sentiment Score Ranges</div>
                    <div class="card-content">
                        <table class="styled-table">
                            <thead>
                                <tr>
                                    <th>Range</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>-1.0 to -0.3</td>
                                    <td>Strongly Negative - Urgent attention required</td>
                                </tr>
                                <tr>
                                    <td>-0.3 to -0.1</td>
                                    <td>Moderately Negative - Concerns need addressing</td>
                                </tr>
                                <tr>
                                    <td>-0.1 to 0.1</td>
                                    <td>Neutral - Balanced public opinion</td>
                                </tr>
                                <tr>
                                    <td>0.1 to 0.3</td>
                                    <td>Moderately Positive - Working well</td>
                                </tr>
                                <tr>
                                    <td>0.3 to 1.0</td>
                                    <td>Strongly Positive - Highly favorable reception</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="card">
                    <div class="card-header">Volatility Thresholds</div>
                    <div class="card-content">
                        <table class="styled-table">
                            <thead>
                                <tr>
                                    <th>Range</th>
                                    <th>Interpretation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>0.0 to 0.1</td>
                                    <td>Low - Consistent public opinion</td>
                                </tr>
                                <tr>
                                    <td>0.1 to 0.25</td>
                                    <td>Medium - Some fluctuation in sentiment</td>
                                </tr>
                                <tr>
                                    <td>Above 0.25</td>
                                    <td>High - Significant shifts in public opinion</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <div class="card-header">Volume Indicators</div>
                <div class="card-content">
                    <ul>
                        <li>Reference volume levels are relative to topic averages</li>
                        <li>Sudden spikes may indicate emerging issues</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <div class="card-header">Action Thresholds</div>
                <div class="card-content">
                    <ul>
                        <li><span class="badge badge-red">High Priority</span>: Sentiment < -0.3 OR Volatility > 0.25</li>
                        <li><span class="badge badge-orange">Medium Priority</span>: Sentiment between -0.3 and -0.1</li>
                        <li><span class="badge badge-blue">Monitor</span>: Neutral sentiment with medium volatility</li>
                        <li><span class="badge badge-green">Maintain</span>: Positive sentiment with low volatility</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: Sentiment Analysis
    with main_tabs[1]:
        st.header("Sentiment Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not filtered_df[['Latitude', 'Longitude']].isnull().all().all():
                fig = px.scatter_geo(filtered_df,
                                    lat='Latitude',
                                    lon='Longitude',
                                    color='sentiment_score',
                                    hover_name='country',
                                    size=np.abs(filtered_df['sentiment_score'])*10,
                                    color_continuous_scale=px.colors.diverging.RdYlGn,
                                    title="Geographical Sentiment Distribution")
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Sentiment Score",
                        tickvals=[-1, -0.3, -0.1, 0.1, 0.3, 1],
                        ticktext=["Very Negative (-1.0)", "Negative (-0.3)", "Slightly Negative (-0.1)", 
                                  "Slightly Positive (0.1)", "Positive (0.3)", "Very Positive (1.0)"]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('<p class="range-info">Bubble size indicates sentiment intensity. Color indicates sentiment direction (red=negative, green=positive).</p>', unsafe_allow_html=True)
        
        with col2:
            if 'sentiment_label' in filtered_df.columns:
                sentiment_counts = filtered_df['sentiment_label'].value_counts()
                fig = px.pie(sentiment_counts,
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            hole=0.3,
                            color=sentiment_counts.index,
                            color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in sentiment_counts.index:
                        count = sentiment_counts[sentiment]
                        pct = (count / sentiment_counts.sum()) * 100
                        st.markdown(f'<p class="range-info">{sentiment.capitalize()}: {count} mentions ({pct:.1f}%)</p>', unsafe_allow_html=True)

    # TAB 3: Trend Insights
    with main_tabs[2]:
        st.header("Trend Insights")
        temp_df = filtered_df.copy()
        if 'created_utc' in temp_df.columns:
            temp_df['date'] = pd.to_datetime(temp_df['created_utc'], unit='s', errors='coerce')
            temp_df = temp_df.dropna(subset=['date'])

            if len(temp_df) > 5:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_avg = temp_df['sentiment_score'].mean()
                    if sentiment_avg < -0.3:
                        sentiment_trend = "Strongly Negative"
                        delta_color = "inverse"
                    elif sentiment_avg < -0.1:
                        sentiment_trend = "Moderately Negative"
                        delta_color = "inverse"
                    elif sentiment_avg < 0.1:
                        sentiment_trend = "Neutral"
                        delta_color = "off"
                    elif sentiment_avg < 0.3:
                        sentiment_trend = "Moderately Positive"
                        delta_color = "normal"
                    else:
                        sentiment_trend = "Strongly Positive"
                        delta_color = "normal"
                    st.metric(
                        "Overall Sentiment", 
                        sentiment_trend,
                        f"{sentiment_avg:.2f} points",
                        delta_color=delta_color
                    )
                    st.markdown('<p class="range-info">Range: -1.0 (very negative) to 1.0 (very positive)</p>', unsafe_allow_html=True)
                
                with col2:
                    volatility = temp_df['sentiment_score'].std()
                    if volatility < 0.1:
                        stability = "High"
                        volatility_class = "Low"
                    elif volatility < 0.25:
                        stability = "Medium"
                        volatility_class = "Medium"
                    else:
                        stability = "Low"
                        volatility_class = "High"
                    st.metric(
                        "Opinion Stability", 
                        stability,
                        f"{volatility_class} volatility: {volatility:.2f}"
                    )
                    st.markdown('<p class="range-info">Volatility ranges: Low (0.0-0.1), Medium (0.1-0.25), High (>0.25)</p>', unsafe_allow_html=True)
                
                with col3:
                    if 'emotion' in temp_df.columns and not temp_df['emotion'].isnull().all():
                        top_emotion = temp_df['emotion'].value_counts().index[0]
                        emotion_pct = (temp_df['emotion'].value_counts().iloc[0] / len(temp_df)) * 100
                        if emotion_pct > 75:
                            prevalence = "Dominant"
                        elif emotion_pct > 50:
                            prevalence = "Major"
                        elif emotion_pct > 25:
                            prevalence = "Common"
                        else:
                            prevalence = "Present"
                        st.metric(
                            "Primary Emotion", 
                            top_emotion,
                            f"{prevalence} ({emotion_pct:.1f}%)"
                        )
                        st.markdown('<p class="range-info">Prevalence: Dominant (>75%), Major (>50%), Common (>25%), Present (<25%)</p>', unsafe_allow_html=True)
                    else:
                        st.metric("Primary Emotion", "N/A", "No emotion data")

                # Narrative Insights
                st.markdown("### Narrative Insights")
                if 'category' in temp_df.columns and not temp_df['category'].isnull().all():
                    category_sentiments = {}
                    for category in selected_categories:
                        cat_data = temp_df[temp_df['category'] == category]
                        if len(cat_data) > 0:
                            category_sentiments[category] = cat_data['sentiment_score'].mean()
                    
                    if category_sentiments:
                        worst_category = min(category_sentiments.items(), key=lambda x: x[1])
                        best_category = max(category_sentiments.items(), key=lambda x: x[1])
                        
                        if worst_category[1] < -0.3:
                            st.error(f"🚨 **High Priority Issue:** The **{worst_category[0]}** category shows strongly negative sentiment (avg: {worst_category[1]:.2f}).")
                        elif worst_category[1] < -0.1:
                            st.warning(f"⚠️ **Medium Priority Issue:** The **{worst_category[0]}** category shows moderately negative sentiment (avg: {worst_category[1]:.2f}).")
                        
                        if best_category[1] > 0.3:
                            st.success(f"✅ **Major Strength:** The **{best_category[0]}** category shows strongly positive sentiment (avg: {best_category[1]:.2f}).")
                        elif best_category[1] > 0.1:
                            st.info(f"💡 **Positive Area:** The **{best_category[0]}** category shows moderately positive sentiment (avg: {best_category[1]:.2f}).")

                # Recommendations
                st.markdown("### Recommended Actions")
                action_items = []
                
                if volatility >= 0.25:
                    action_items.append("🔍 **HIGH PRIORITY:** High sentiment volatility (>{0.25}) indicates public opinion is frequently shifting. Implement more consistent messaging and monitor more frequently.")
                elif volatility >= 0.1:
                    action_items.append("⚠️ **MEDIUM PRIORITY:** Medium volatility (0.1-0.25) suggests some opinion fluctuation. Analyze triggers for sentiment changes.")
                
                if sentiment_avg < -0.3:
                    action_items.append("🚨 **HIGH PRIORITY:** Strongly negative sentiment (<-0.3) requires immediate action. Develop a comprehensive response strategy.")
                elif sentiment_avg < -0.1:
                    action_items.append("⚠️ **MEDIUM PRIORITY:** Moderately negative sentiment (-0.3 to -0.1) indicates issues that need addressing in the near term.")
                elif sentiment_avg > 0.3:
                    action_items.append("✅ **LEVERAGE STRENGTHS:** Strongly positive sentiment (>0.3) shows areas of success that can be highlighted and expanded.")
                elif sentiment_avg > 0.1:
                    action_items.append("💡 **MAINTAIN COURSE:** Moderately positive sentiment (0.1 to 0.3) suggests current strategies are working well.")
                else:
                    action_items.append("📊 **MONITOR CLOSELY:** Neutral sentiment (-0.1 to 0.1) indicates balanced public opinion. Watch for shifts in either direction.")
                
                if 'category' in locals() and 'category_sentiments' in locals() and category_sentiments:
                    negative_cats = [cat for cat, score in category_sentiments.items() if score < -0.3]
                    moderate_neg_cats = [cat for cat, score in category_sentiments.items() if -0.3 <= score < -0.1]
                    positive_cats = [cat for cat, score in category_sentiments.items() if score > 0.3]
                    
                    if negative_cats:
                        action_items.append(f"🚨 **HIGH PRIORITY CATEGORIES:** Address significant concerns in: {', '.join(negative_cats)}")
                    
                    if moderate_neg_cats:
                        action_items.append(f"⚠️ **MEDIUM PRIORITY CATEGORIES:** Develop improvement plans for: {', '.join(moderate_neg_cats)}")
                        
                    if positive_cats:
                        action_items.append(f"💡 **STRENGTH CATEGORIES:** Leverage positive perception in: {', '.join(positive_cats)}")
                
                for item in action_items:
                    st.markdown(f"{item}")

                # Executive Summary
                st.markdown("---")
                st.markdown("### Executive Summary")
                if sentiment_avg < -0.3:
                    sentiment_summary = "strongly negative"
                    sentiment_action = "requires immediate attention"
                elif sentiment_avg < -0.1:
                    sentiment_summary = "moderately negative"
                    sentiment_action = "needs improvement"
                elif sentiment_avg < 0.1:
                    sentiment_summary = "neutral"
                    sentiment_action = "requires monitoring"
                elif sentiment_avg < 0.3:
                    sentiment_summary = "moderately positive"
                    sentiment_action = "is progressing well"
                else:
                    sentiment_summary = "strongly positive"
                    sentiment_action = "is excelling"
                    
                if volatility < 0.1:
                    volatility_summary = "consistently"
                    volatility_action = "stable messaging is effective"
                elif volatility < 0.25:
                    volatility_summary = "with some fluctuation"
                    volatility_action = "more consistent messaging may help"
                else:
                    volatility_summary = "with significant fluctuations"
                    volatility_action = "a more unified communication strategy is required"
                
                summary_text = f"Public sentiment is currently **{sentiment_summary}** ({sentiment_avg:.2f}) **{volatility_summary}** (volatility: {volatility:.2f}). "
                summary_text += f"The data suggests that public opinion {sentiment_action} and {volatility_action}."
                st.markdown(summary_text)
            else:
                st.info("📊 **Insufficient Data:** More data points are needed for trend analysis. Current dataset is too small for reliable insights.")
        else:
            st.warning("Missing timestamp data needed for trend analysis")

    # TAB 4: Actionable Insights
    with main_tabs[3]:
        st.header("Actionable Insights by Category")
        if not filtered_df.empty:
            insights = []
            
            for category in selected_categories:
                cat_df = filtered_df[filtered_df['category'] == category]
                if len(cat_df) == 0:
                    continue
                
                avg_sentiment = cat_df['sentiment_score'].mean()
                sentiment_level = "Strongly Positive" if avg_sentiment > 0.3 else \
                                  "Moderately Positive" if avg_sentiment > 0.1 else \
                                  "Neutral" if avg_sentiment >= -0.1 else \
                                  "Moderately Negative" if avg_sentiment >= -0.3 else "Strongly Negative"
                
                top_emotion = cat_df['emotion'].mode()[0] if not cat_df['emotion'].isnull().all() else "N/A"
                mention_pct = (len(cat_df) / len(filtered_df)) * 100
                
                insight = f"""
                **{category}**
                - Sentiment: {avg_sentiment:.2f} ({sentiment_level})
                - Dominant Emotion: {top_emotion}
                - Volume: {len(cat_df)} mentions ({mention_pct:.1f}%)
                - Priority: {"High" if avg_sentiment < -0.3 else "Medium" if avg_sentiment < -0.1 else "Low"}
                """
                insights.append(insight)
            
            cols = st.columns(len(insights))
            for col, insight in zip(cols, insights):
                with col:
                    st.markdown(insight, unsafe_allow_html=True)
        else:
            st.warning("No data available for current filters")

    # TAB 5: Documentation
    with main_tabs[4]:
        st.header("Documentation")
        st.markdown("""
        ### Data Sources
        - **Reddit**: Public forums and discussions
        - **News Articles**: Major news outlets and publications
        - **RSS Feeds**: Aggregated content from various sources
        - **Public Forums**: Community discussions and feedback
        
        ### Sentiment Analysis Methodology
        - **Sentiment Scale**: -1.0 (strongly negative) to 1.0 (strongly positive)
        - **Volatility Scale**: 0.0 (stable) to 1.0 (highly volatile)
        - **Emotion Detection**: Based on text analysis and natural language processing
        
        ### Interpretation Guide
        - **Sentiment Scores**: Ranges from -1.0 to 1.0, with specific thresholds for negative, neutral, and positive sentiment
        - **Volatility**: Measures the consistency of public opinion over time
        - **Volume**: Indicates the relative amount of discussion on a topic
        
        ### Recommended Actions
        - **High Priority**: Issues with strongly negative sentiment or high volatility
        - **Medium Priority**: Issues with moderately negative sentiment or medium volatility
        - **Monitor**: Neutral sentiment with medium volatility
        - **Maintain**: Positive sentiment with low volatility
        """)

    # Footer
    st.markdown("---")
    st.caption("Sources: Reddit, News Articles, RSS Feeds, Public Forums")
    st.caption("Sentiment Scale: -1.0 (strongly negative) to 1.0 (strongly positive) | Volatility Scale: 0.0 (stable) to 1.0 (highly volatile)")



if __name__ == "__main__":
    run_dashboard()