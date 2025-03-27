import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def load_data():
    try:
        # Historical scores data
        historical_scores = pd.read_csv('dashboard_data/historical_scores.csv')
        forecasts = pd.read_csv('dashboard_data/forecasts.csv')
        country_summary = pd.read_csv('dashboard_data/country_summary.csv')
        factor_correlations = pd.read_csv('dashboard_data/factor_correlations.csv')
        top_countries = pd.read_csv('dashboard_data/top_countries.csv')
        bottom_countries = pd.read_csv('dashboard_data/bottom_countries.csv')
        
        # Load the world bank data
        world_bank_data = pd.read_csv('updated_world_bank_195_countries_complete.csv')
        
        # Add current rank to country summary
        country_summary['Current_Rank'] = country_summary['Latest_Score'].rank(method='dense', ascending=False).astype(int)
        
        def country_forecast_page(data):
            st.header("Country Livability Score Forecasts")
            
            # Country selection
            countries = sorted(data['forecasts']['Country'].unique())
            selected_country = st.selectbox("Select a Country", countries)
            
            # Filter data for selected country
            country_historical = data['historical_scores'][data['historical_scores']['Country'] == selected_country]
            country_forecast = data['forecasts'][data['forecasts']['Country'] == selected_country]
            
            # Create forecast plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=country_historical['Year'], 
                y=country_historical['Livability_Score'], 
                mode='lines+markers', 
                name='Historical Scores',
                line=dict(color='blue')
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=country_forecast['Year'], 
                y=country_forecast['Livability_Score'], 
                mode='lines+markers', 
                name='Forecast Scores',
                line=dict(color='red', dash='dot')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=country_forecast['Year'], 
                y=country_forecast['Upper_Bound'], 
                mode='lines', 
                name='Upper Bound',
                line=dict(width=0),
                showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=country_forecast['Year'], 
                y=country_forecast['Lower_Bound'], 
                mode='lines', 
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f'Livability Score Forecast for {selected_country}',
                xaxis_title='Year',
                yaxis_title='Livability Score',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional country insights
            country_info = data['country_summary'][data['country_summary']['Country'] == selected_country].iloc[0]
            
            # Create two columns for cards
            col1, col2 = st.columns(2)
            
            # Card 1: Country Metrics
            with col1:
                st.markdown(f"""
                ### 📊 Country Performance Metrics
                <div style='border: 1px solid #e6e6e6; border-radius: 10px; padding: 15px; background-color: #f9f9f9;'>
                    <h4 style='margin-bottom: 10px;'>Key Performance Indicators</h4>
                    <p><strong>Current Rank:</strong> {country_info['Current_Rank']}</p>
                    <p><strong>Latest Score:</strong> {country_info['Latest_Score']:.3f}</p>
                    <p><strong>2027 Forecast:</strong> {country_info['Forecast_2027']:.3f}</p>
                    <p style='color: {"green" if country_info["Percent_Change"] > 0 else "red"}'>
                        <strong>Percent Change:</strong> {country_info['Percent_Change']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Fetch latest year's data for the selected country
            latest_year_data = world_bank_data[
                (world_bank_data['Country'] == selected_country) & 
                (world_bank_data['Year'] == 2022)
            ]
            
            # Card 2: Latest Country Indicators
            with col2:
                if not latest_year_data.empty:
                    st.markdown(f"""
                    ### 🌐 Latest Country Indicators (2022)
                    <div style='border: 1px solid #e6e6e6; border-radius: 10px; padding: 15px; background-color: #f9f9f9;'>
                        <h4 style='margin-bottom: 10px;'>Key Socio-Economic Metrics</h4>
                        <p><strong>Electricity Access:</strong> {latest_year_data['Electricity access (% of population)'].values[0]:.2f}%</p>
                        <p><strong>Forest Area:</strong> {latest_year_data['Forest area (% of land area)'].values[0]:.2f}%</p>
                        <p><strong>GDP per Capita:</strong> ${latest_year_data['GDP per capita (current US$)'].values[0]:,.2f}</p>
                        <p><strong>Life Expectancy:</strong> {latest_year_data['Life expectancy at birth (years)'].values[0]:.2f} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    ### 🌐 Latest Country Indicators
                    <div style='border: 1px solid #e6e6e6; border-radius: 10px; padding: 15px; background-color: #f9f9f9;'>
                        <p>No recent indicators available for the selected country.</p>
                    </div>
                    """, unsafe_allow_html=True)

        def country_comparison_page(data):
            st.header("Country Comparison")
            
            # Get all countries
            all_countries = sorted(data['country_summary']['Country'].unique())
            
            # Multiselect for countries
            selected_countries = st.multiselect(
                "Select Countries to Compare", 
                all_countries, 
                default=all_countries[:5]  # Default to first 5 countries
            )
            
            # Filter data for selected countries
            comparison_data = data['country_summary'][
                data['country_summary']['Country'].isin(selected_countries)
            ].copy()
            
            # Create comparison columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.subheader("Country")
                for country in comparison_data['Country']:
                    st.write(country)
            
            with col2:
                st.subheader("Current Rank")
                for rank in comparison_data['Current_Rank']:
                    st.write(str(rank))
            
            with col3:
                st.subheader("Latest Score")
                for score in comparison_data['Latest_Score']:
                    st.write(f"{score:.3f}")
            
            with col4:
                st.subheader("2027 Forecast")
                for forecast in comparison_data['Forecast_2027']:
                    st.write(f"{forecast:.3f}")
            
            # Comparative visualization
            st.subheader("Comparative Livability Scores")
            
            # Bar chart for latest scores and forecasts
            fig_comparison = go.Figure()
            
            # Latest Scores
            fig_comparison.add_trace(go.Bar(
                x=comparison_data['Country'],
                y=comparison_data['Latest_Score'],
                name='Latest Score',
                marker_color='blue'
            ))
            
            # 2027 Forecasts
            fig_comparison.add_trace(go.Bar(
                x=comparison_data['Country'],
                y=comparison_data['Forecast_2027'],
                name='2027 Forecast',
                marker_color='green'
            ))
            
            fig_comparison.update_layout(
                title='Comparison of Latest Scores and 2027 Forecasts',
                xaxis_title='Country',
                yaxis_title='Livability Score',
                barmode='group'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

        def top_bottom_countries_page(data):
            st.header("Country Livability Rankings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Countries")
                top_df = data['top_countries'].copy()
                top_df['Current_Rank'] = top_df['Forecast_2027'].rank(method='dense', ascending=False).astype(int)
                
                fig_top = px.bar(
                    top_df, 
                    x='Country', 
                    y='Forecast_2027', 
                    title='Top 10 Countries by 2027 Forecast',
                    labels={'Forecast_2027': 'Livability Score'},
                    color='Percent_Change',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col2:
                st.subheader("Bottom 10 Countries")
                bottom_df = data['bottom_countries'].copy()
                bottom_df['Current_Rank'] = bottom_df['Forecast_2027'].rank(method='dense', ascending=True).astype(int)
                
                fig_bottom = px.bar(
                    bottom_df, 
                    x='Country', 
                    y='Forecast_2027', 
                    title='Bottom 10 Countries by 2027 Forecast',
                    labels={'Forecast_2027': 'Livability Score'},
                    color='Percent_Change',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_bottom, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Country Comparison")
            comparison_df = pd.concat([top_df, bottom_df])
            comparison_df = comparison_df.style.format({
                'Latest_Score': '{:.3f}', 
                'Forecast_2027': '{:.3f}', 
                'Percent_Change': '{:.2f}%',
                'Current_Rank': '{}'
            })
            st.dataframe(comparison_df)

        def factor_correlations_page(data):
            st.header("Factor Correlations with Livability Score")
            
            correlations = data['factor_correlations']
            
            # Create correlation bar plot
            fig = px.bar(
                correlations, 
                x='Factor', 
                y='Correlation', 
                title='Correlation of Factors with Livability Score',
                labels={'Correlation': 'Correlation Strength'},
                color='Correlation',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed correlation explanation
            st.markdown("""
            ### Interpreting Correlations
            - Positive correlation (closer to 1.0): As the factor increases, livability score tends to increase
            - Negative correlation (closer to -1.0): As the factor increases, livability score tends to decrease
            - Near zero: Weak or no linear relationship with livability score
            """)
            
            # Detailed table
            st.dataframe(correlations.style.format({'Correlation': '{:.3f}'}))

        def about_model_page():
            st.header("About the Livability Model")
            
            st.markdown("""
            ### Methodology
            This predictive model assesses global livability by analyzing multiple socio-economic and environmental indicators:
            
            #### Key Indicators
            - Access to safe drinking water
            - Forest area
            - GDP per capita
            - Life expectancy
            - Literacy rate
            - Unemployment rate
            - Electricity access
            - Renewable energy consumption
            - CO2 emissions
            - Inflation rate
            
            #### Modeling Approach
            - Data preprocessing and normalization
            - Missing value imputation
            - Time series forecasting using Facebook Prophet
            - Evaluation using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
            
            ### Forecasting Details
            - Forecast Period: 2023-2027
            - Methodology ensures robust predictions by:
              * Using historical data trends
              * Accounting for potential changes
              * Providing confidence intervals
            
            ### Limitations
            - Predictions based on historical data
            - Does not account for unexpected global events
            - Model accuracy depends on data quality and completeness
            """)

        def main():
            # Title and introduction
            st.title("🌍 Global Livability Forecast (2023-2027)")
            st.markdown("""
            This dashboard provides insights into the predicted livability scores across countries, 
            based on key socio-economic and environmental indicators.
            """)
            
            # Sidebar for navigation
            page = st.sidebar.selectbox("Select View", [
                "Country Forecasts", 
                "Country Comparison",
                "Top & Bottom Countries", 
                "Factor Correlations", 
                "About the Model"
            ])
            
            if page == "Country Forecasts":
                country_forecast_page(dashboard_data)
            elif page == "Country Comparison":
                country_comparison_page(dashboard_data)
            elif page == "Top & Bottom Countries":
                top_bottom_countries_page(dashboard_data)
            elif page == "Factor Correlations":
                factor_correlations_page(dashboard_data)
            elif page == "About the Model":
                about_model_page()

        # Create a dashboard data dictionary
        dashboard_data = {
            'historical_scores': historical_scores,
            'forecasts': forecasts,
            'country_summary': country_summary,
            'factor_correlations': factor_correlations,
            'top_countries': top_countries,
            'bottom_countries': bottom_countries
        }

        # Additional configuration for the page
        main()
        
        return dashboard_data
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Optional: If you want to run the script directly
if __name__ == "__main__":
    load_data()