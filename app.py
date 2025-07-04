# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Investment Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #003087;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #003087;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .cluster-description {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #c2d6f0;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #003087;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        border: none;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0052cc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .welcome-box {
        background: linear-gradient(135deg, #003087 0%, #1f77b4 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class InvestmentSegmentationApp:
    """
    Investment Customer Segmentation Streamlit Application
    """
    
    def __init__(self):
        self.models_loaded = False
        self.load_models_and_data()
        
    def load_models_and_data(self):
        """Load trained models and preprocessors"""
        try:
            self.scaler = joblib.load("models/scaler.pkl")
            self.kmeans_model = joblib.load("models/kmeans_model.pkl")
            self.gmm_model = joblib.load("models/gmm_model.pkl")
            self.label_encoders = joblib.load("models/label_encoders.pkl")
            
            if os.path.exists("models/model_metadata.json"):
                with open("models/model_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
            
            if os.path.exists("outputs/cluster_profiles.json"):
                with open("outputs/cluster_profiles.json", 'r') as f:
                    self.cluster_profiles = json.load(f)
            else:
                self.cluster_profiles = {}
            
            if os.path.exists("outputs/business_insights.json"):
                with open("outputs/business_insights.json", 'r') as f:
                    self.business_insights = json.load(f)
            else:
                self.business_insights = []
            
            if os.path.exists("outputs/clustered_dataset.csv"):
                self.df_clustered = pd.read_csv("outputs/clustered_dataset.csv")
            else:
                self.df_clustered = pd.DataFrame()
            
            self.models_loaded = True
            logger.info("Models and data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please run the training pipeline first using train_models.py")
            self.models_loaded = False
    
    def calculate_risk_score(self, age, saving, freq, duration):
        """Calculate risk score based on customer inputs"""
        score = (age * 0.3) + (saving / 8000) + (freq * 1.2) - (duration * 0.1)
        return max(0, min(100, round(score, 1)))
    
    def calculate_roi(self, risk_score, freq, inv_type):
        """Calculate expected ROI based on risk score and investment type"""
        base_roi = 5
        risk_bonus = risk_score / 20
        freq_bonus = freq * 0.1
        
        inv_type_bonus_map = {
            'stocks': 5, 'mutual': 4, 'crypto': 8, 'gold': 2,
            'fd': 1.5, 'ppf': 2, 'bonds': 2.5, 'real_estate': 3,
            'etf': 4, 'nps': 2
        }
        inv_type_bonus = inv_type_bonus_map.get(inv_type, 2)
        
        return round(base_roi + risk_bonus + freq_bonus + inv_type_bonus, 2)
    
    def prepare_input_features(self, age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type):
        """Prepare input features for prediction"""
        numerical_features = pd.DataFrame({
            'age': [age],
            'saving': [saving],
            'net_worth': [net_worth],
            'risk_score': [risk_score],
            'tenure': [tenure],
            'duration': [duration],
            'freq': [freq],
            'roi': [roi]
        })
        
        categorical_features = pd.DataFrame()
        
        if 'goal' in self.label_encoders:
            try:
                goal_encoded = self.label_encoders['goal'].transform([goal])[0]
                categorical_features['goal_encoded'] = [goal_encoded]
            except ValueError:
                categorical_features['goal_encoded'] = [0]
        
        if 'inv_type' in self.label_encoders:
            try:
                inv_type_encoded = self.label_encoders['inv_type'].transform([inv_type])[0]
                categorical_features['inv_type_encoded'] = [inv_type_encoded]
            except ValueError:
                categorical_features['inv_type_encoded'] = [0]
        
        if categorical_features.empty:
            input_features = numerical_features
        else:
            input_features = pd.concat([numerical_features, categorical_features], axis=1)
        
        return input_features
    
    def predict_cluster(self, input_features):
        """Predict cluster for input features"""
        try:
            scaled_features = self.scaler.transform(input_features)
            kmeans_cluster = self.kmeans_model.predict(scaled_features)[0]
            gmm_cluster = self.gmm_model.predict(scaled_features)[0]
            return kmeans_cluster, gmm_cluster
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None, None
    
    def get_cluster_description(self, cluster_id, method='kmeans'):
        """Get dynamic description for a cluster based on cluster_id"""
        cluster_descriptions = {
            0: {
                'name': '🛡️ Conservative Investors',
                'description': 'Low-risk, stability-focused investors who prioritize capital preservation over high returns.',
                'characteristics': ['Lower risk tolerance', 'Stable income', 'Long-term focused', 'Prefer guaranteed returns']
            },
            1: {
                'name': '📊 Balanced Investors',
                'description': 'Moderate risk-taking investors seeking balanced growth with reasonable security.',
                'characteristics': ['Moderate risk tolerance', 'Diversified portfolio', 'Mixed investment horizon', 'Growth with safety']
            },
            2: {
                'name': '🚀 Aggressive Investors',
                'description': 'High-risk, high-reward seeking investors comfortable with market volatility.',
                'characteristics': ['High risk tolerance', 'Growth-oriented', 'Tech-savvy', 'Selective active trading']
            },
            3: {
                'name': '💼 Wealth Builders',
                'description': 'Long-term wealth accumulation focused investors with substantial assets.',
                'characteristics': ['High net worth', 'Strategic planning', 'Wealth preservation', 'Diversified long-term investments']
            }
        }
        
        return cluster_descriptions.get(cluster_id, {
            'name': f'Cluster {cluster_id}',
            'description': 'Investment profile not fully characterized',
            'characteristics': ['Mixed characteristics']
        })

    def generate_recommendations(self, cluster_id, risk_score, goal, inv_type):
        """Generate tailored investment recommendations based on cluster, risk, goal, and investment type"""
        recommendations = []
        
        # Cluster-specific insights
        cluster_insight = next((insight for insight in self.business_insights if insight['cluster_id'] == cluster_id), None)
        if cluster_insight:
            recommendations.append(f"**Primary Strategy:** {cluster_insight['business_strategy']}")

        # Risk-based recommendations
        if risk_score < 25:
            recommendations.extend([
                "Focus on low-risk options like PPF, Fixed Deposits, or Government Bonds.",
                "Prioritize capital preservation with tax-saving instruments like ELSS.",
                "Maintain a liquid emergency fund for stability."
            ])
        elif 25 <= risk_score < 50:
            recommendations.extend([
                "Diversify with balanced mutual funds or ETFs.",
                "Consider hybrid funds for a mix of equity and debt.",
                "Explore mid-term investment horizons for steady growth."
            ])
        else:
            recommendations.extend([
                "Target high-growth assets such as stocks or cryptocurrencies.",
                "Use Systematic Investment Plans (SIPs) for disciplined equity exposure.",
                "Stay updated on market trends for strategic opportunities."
            ])

        # Goal-specific recommendations
        goal_recommendations = {
            'retirement': ['Maximize EPF/NPS contributions', 'Invest in long-term equity index funds', 'Balance with debt for stability'],
            'wealth': ['Focus on equity and real estate growth', 'Consider international diversification', 'Rebalance annually'],
            'education': ['Use target-date funds aligned with timelines', 'Invest in debt for safety', 'Plan for inflation adjustments'],
            'vacation': ['Opt for short-term debt or liquid funds', 'Set aside a fixed monthly amount', 'Avoid high-risk volatility'],
            'emergency': ['Build a high-yield savings account', 'Use liquid mutual funds', 'Keep funds accessible without penalties']
        }
        if goal in goal_recommendations:
            recommendations.extend(goal_recommendations[goal])

        # Cluster-specific tailoring
        if cluster_id == 0:  # Conservative Investors
            recommendations.append("Prioritize fixed-income securities and avoid speculative investments.")
        elif cluster_id == 1:  # Balanced Investors
            recommendations.append("Maintain a 50/50 equity-debt ratio for balanced risk management.")
        elif cluster_id == 2:  # Aggressive Investors
            recommendations.append("Leverage high-volatility opportunities like tech stocks or crypto, but set stop-loss limits.")
        elif cluster_id == 3:  # Wealth Builders
            recommendations.append("Focus on asset allocation across real estate and equity for long-term wealth growth.")

        # Investment type alignment with expanded options
        inv_type_strategy = {
            'stocks': "Increase equity exposure with a focus on blue-chip or growth stocks.",
            'crypto': "Limit to 5-10% of portfolio due to high volatility.",
            'mutual': "Opt for diversified or sector-specific funds.",
            'gold': "Use as a hedge against inflation and market downturns.",
            'fd': "Secure short-term savings with competitive rates.",
            'ppf': "Leverage tax benefits for long-term goals.",
            'bonds': "Invest in corporate or government bonds for steady income.",
            'real_estate': "Consider REITs or property investments for diversification and appreciation.",
            'etf': "Use exchange-traded funds for low-cost, broad market exposure.",
            'nps': "Maximize National Pension System for retirement planning with tax benefits."
        }
        recommendations.append(inv_type_strategy.get(inv_type, "Review your investment type for optimal alignment."))

        return recommendations
    
    def create_interactive_charts(self):
        """Create 10 useful interactive charts using Plotly"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for visualization")
            return
        
        # Create subplot layout (5 rows x 2 columns)
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                "Risk Score vs ROI by Cluster",
                "Age vs Net Worth Distribution",
                "Investment Type Distribution",
                "Cluster Distribution",
                "Cluster Comparison: Risk Score",
                "Cluster Comparison: ROI",
                "Cluster Comparison: Age",
                "Cluster Comparison: Net Worth",
                "Goal Distribution by Cluster",
                "Feature Correlation Heatmap"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "box"}, {"type": "box"}],
                [{"type": "box"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.1
        )
        
        # 1. Risk vs ROI scatter plot
        scatter = go.Scatter(
            x=self.df_clustered['risk_score'],
            y=self.df_clustered['roi'],
            mode='markers',
            marker=dict(
                color=self.df_clustered['kmeans_cluster'],
                colorscale='Viridis',
                showscale=True,
                size=10,
                opacity=0.7
            ),
            text=self.df_clustered.apply(
                lambda x: f"Age: {x['age']}<br>Goal: {x['goal']}<br>Cluster: {x['kmeans_cluster']}", axis=1),
            hoverinfo='text'
        )
        fig.add_trace(scatter, row=1, col=1)
        
        # 2. Age vs Net Worth histogram
        hist = go.Histogram2d(
            x=self.df_clustered['age'],
            y=self.df_clustered['net_worth'] / 100000,
            colorscale='Blues',
            nbinsx=20,
            nbinsy=20,
            hoverinfo='x+y+z'
        )
        fig.add_trace(hist, row=1, col=2)
        
        # 3. Investment Type Distribution
        inv_type_counts = self.df_clustered['inv_type'].value_counts().sort_values(ascending=False)
        bar = go.Bar(
            x=inv_type_counts.index,
            y=inv_type_counts.values,
            marker=dict(color=inv_type_counts.values, colorscale='Portland')
        )
        fig.add_trace(bar, row=2, col=1)
        
        # 4. Cluster Distribution
        cluster_counts = self.df_clustered['kmeans_cluster'].value_counts().sort_index()
        pie = go.Pie(
            labels=[self.get_cluster_description(i)['name'] for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Plotly)
        )
        fig.add_trace(pie, row=2, col=2)
        
        # 5-8. Cluster Comparisons: Risk Score, ROI, Age, Net Worth (Box Plots)
        for cluster_id in sorted(self.df_clustered['kmeans_cluster'].unique()):
            cluster_data = self.df_clustered[self.df_clustered['kmeans_cluster'] == cluster_id]
            cluster_name = self.get_cluster_description(cluster_id)['name']
            
            # Risk Score Box Plot
            box_risk = go.Box(
                y=cluster_data['risk_score'],
                name=cluster_name,
                marker_color=px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)],
                showlegend=False
            )
            fig.add_trace(box_risk, row=3, col=1)
            
            # ROI Box Plot
            box_roi = go.Box(
                y=cluster_data['roi'],
                name=cluster_name,
                marker_color=px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)],
                showlegend=False
            )
            fig.add_trace(box_roi, row=3, col=2)
            
            # Age Box Plot
            box_age = go.Box(
                y=cluster_data['age'],
                name=cluster_name,
                marker_color=px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)],
                showlegend=False
            )
            fig.add_trace(box_age, row=4, col=1)
            
            # Net Worth Box Plot
            box_net_worth = go.Box(
                y=cluster_data['net_worth'] / 100000,
                name=cluster_name,
                marker_color=px.colors.qualitative.Plotly[cluster_id % len(px.colors.qualitative.Plotly)],
                showlegend=False
            )
            fig.add_trace(box_net_worth, row=4, col=2)
        
        # 9. Goal Distribution by Cluster (Stacked Bar)
        goal_pivot = self.df_clustered.groupby(['kmeans_cluster', 'goal']).size().unstack(fill_value=0)
        for goal in goal_pivot.columns:
            fig.add_trace(
                go.Bar(
                    x=[self.get_cluster_description(i)['name'] for i in goal_pivot.index],
                    y=goal_pivot[goal],
                    name=goal.capitalize(),
                    marker_color=px.colors.qualitative.Plotly[goal_pivot.columns.get_loc(goal) % len(px.colors.qualitative.Plotly)]
                ),
                row=5, col=1
            )
        
        # 10. Feature Correlation Heatmap
        numerical_cols = ['age', 'saving', 'net_worth', 'risk_score', 'tenure', 'duration', 'freq', 'roi']
        corr_matrix = self.df_clustered[numerical_cols].corr()
        heatmap = go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            showscale=True
        )
        fig.add_trace(heatmap, row=5, col=2)
        
        # Update layout
        fig.update_layout(
            height=1800,
            showlegend=True,
            title_text="Customer Segmentation Analytics Dashboard",
            title_x=0.5,
            title_font=dict(size=20, color='#003087'),
            barmode='stack'  # For stacked bar chart
        )
        fig.update_xaxes(title_text="Risk Score", row=1, col=1)
        fig.update_yaxes(title_text="ROI (%)", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Net Worth (Lakhs)", row=1, col=2)
        fig.update_xaxes(title_text="Investment Type", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=3, col=1)
        fig.update_yaxes(title_text="Risk Score", row=3, col=1)
        fig.update_xaxes(title_text="Cluster", row=3, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=3, col=2)
        fig.update_xaxes(title_text="Cluster", row=4, col=1)
        fig.update_yaxes(title_text="Age", row=4, col=1)
        fig.update_xaxes(title_text="Cluster", row=4, col=2)
        fig.update_yaxes(title_text="Net Worth (Lakhs)", row=4, col=2)
        fig.update_xaxes(title_text="Cluster", row=5, col=1)
        fig.update_yaxes(title_text="Count", row=5, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_cluster_analysis(self):
        """Display comprehensive cluster analysis with improved visuals"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for analysis")
            return
        
        st.subheader("📊 Cluster Analysis Overview")
        
        for cluster_id in sorted(self.df_clustered['kmeans_cluster'].unique()):
            cluster_data = self.df_clustered[self.df_clustered['kmeans_cluster'] == cluster_id]
            cluster_desc = self.get_cluster_description(cluster_id)
            
            with st.expander(f"{cluster_desc['name']} (n={len(cluster_data)})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Key Metrics**")
                    metrics = [
                        ("Average Age", f"{cluster_data['age'].mean():.1f} years"),
                        ("Average Risk Score", f"{cluster_data['risk_score'].mean():.1f}"),
                        ("Average ROI", f"{cluster_data['roi'].mean():.1f}%"),
                        ("Average Net Worth", f"₹{cluster_data['net_worth'].mean()/100000:.1f}L")
                    ]
                    for label, value in metrics:
                        st.markdown(f'<div class="metric-card"><strong>{label}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Profile**")
                    st.markdown(f'<div class="cluster-description">{cluster_desc["description"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Characteristics**")
                    for char in cluster_desc['characteristics']:
                        st.markdown(f"✓ {char}")
                
                # Visual: Top goals and investment types
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Top Goals**")
                    top_goals = cluster_data['goal'].value_counts().head(3)
                    fig_goals = go.Figure(data=[
                        go.Bar(x=top_goals.values, y=top_goals.index, orientation='h',
                               marker=dict(color='#003087'))
                    ])
                    fig_goals.update_layout(height=200, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_goals, use_container_width=True)
                
                with col4:
                    st.markdown("**Top Investment Types**")
                    top_inv = cluster_data['inv_type'].value_counts().head(3)
                    fig_inv = go.Figure(data=[
                        go.Bar(x=top_inv.values, y=top_inv.index, orientation='h',
                               marker=dict(color='#1f77b4'))
                    ])
                    fig_inv.update_layout(height=200, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_inv, use_container_width=True)
    
    def customer_segmentation_page(self):
        """Main customer segmentation page with enhanced UI"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">Investment Customer Segmentation 💹</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.error("⚠️ Models not loaded. Please run train_models.py first!")
            return
        
        st.markdown("""
        <div class="welcome-box">
            <h3>Welcome to IBM's Investment Analytics</h3>
            <p>Discover your investment profile and get personalized recommendations powered by advanced machine learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Enter Your Details")
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            col1.header("Personal Details")
            age = col1.number_input("Age", min_value=18, max_value=75, value=30)
            net_worth = col1.number_input("Net Worth (₹)", min_value=10000, max_value=10000000, value=500000, step=50000)
            
            col2.header("Investment Profile")
            saving = col2.number_input("Monthly Saving (₹)", min_value=1000, max_value=100000, value=20000, step=1000)
            tenure = col2.slider("Years of Investing Experience", 0, 30, 5)
            duration = col2.slider("Average Investment Duration (months)", 1, 120, 24)
            freq = col2.slider("Investment Frequency (per year)", 1, 30, 5)
            
            col1.header("Investment Preferences")
            goal = col1.selectbox("Primary Financial Goal",
                                 ['retirement', 'wealth', 'education', 'vacation', 'emergency'])
            inv_type = col2.selectbox("Preferred Investment Type",
                                     ['stocks', 'mutual', 'crypto', 'gold', 'fd', 'ppf', 'bonds', 'real_estate', 'etf', 'nps'])
            
            submit_button = st.form_submit_button("🔍 Analyze My Profile", type="primary")
        
        if submit_button:
            risk_score = self.calculate_risk_score(age, saving, freq, duration)
            roi = self.calculate_roi(risk_score, freq, inv_type)
            
            st.markdown("## 📈 Your Investment Profile")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<div class="metric-card"><strong>Risk Score</strong><br>{risk_score}/100</div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><strong>Expected ROI</strong><br>{roi}%</div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card"><strong>Investment Horizon</strong><br>{duration} months</div>', unsafe_allow_html=True)
            
            input_features = self.prepare_input_features(
                age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal, inv_type
            )
            kmeans_cluster, gmm_cluster = self.predict_cluster(input_features)
            
            if kmeans_cluster is not None:
                cluster_desc = self.get_cluster_description(kmeans_cluster)
                
                st.markdown("## 🎯 Your Investor Profile")
                st.markdown(f'<div class="cluster-description">', unsafe_allow_html=True)
                st.markdown(f"### {cluster_desc['name']}")
                st.markdown(f"**Cluster ID:** {kmeans_cluster}")
                st.markdown(f"**Description:** {cluster_desc['description']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                for i, char in enumerate(cluster_desc['characteristics']):
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"✓ {char}")
                
                st.markdown("## 💡 Personalized Recommendations")
                recommendations = self.generate_recommendations(kmeans_cluster, risk_score, goal, inv_type)
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("🔍 Model Details"):
                    col1, col2 = st.columns(2)
                    col1.metric("KMeans Cluster", kmeans_cluster)
                    col2.metric("GMM Cluster", gmm_cluster)
                    if 'kmeans_silhouette' in self.metadata:
                        st.info(f"Model Quality - KMeans Silhouette Score: {self.metadata['kmeans_silhouette']:.3f}")

    
    def analytics_dashboard_page(self):
        """Analytics dashboard with categorized and enhanced visualizations"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded or self.df_clustered.empty:
            st.warning("⚠️ No data available for analytics. Please run the training pipeline first.")
            return
        
        st.markdown("## 📈 Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Customers", len(self.df_clustered)),
            ("Average Risk Score", f"{self.df_clustered['risk_score'].mean():.1f}"),
            ("Average ROI", f"{self.df_clustered['roi'].mean():.1f}%"),
            ("Clusters Identified", len(self.df_clustered['kmeans_cluster'].unique()))
        ]
        for i, (label, value) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f'<div class="stat-card"><strong>{label}</strong><br>{value}</div>', unsafe_allow_html=True)
        
        st.markdown("## 📊 Investment Insights")
        self.create_interactive_charts()
        
        self.display_cluster_analysis()
        
        if self.business_insights:
            st.markdown("## 💼 Strategic Insights")
            for insight in self.business_insights:
                with st.expander(f"Cluster {insight['cluster_id']}: {insight['segment_type']}"):
                    st.markdown(f'<div class="metric-card"><strong>Size:</strong> {insight["size"]} customers</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><strong>Strategy:</strong> {insight["business_strategy"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Key Characteristics**")
                    for key, value in insight['key_characteristics'].items():
                        st.markdown(f"• {key.replace('_', ' ').title()}: {value}")

    def create_interactive_charts(self):
        """Create categorized, larger, and easily understandable interactive charts using Plotly"""
        if self.df_clustered.empty:
            st.warning("No clustered data available for visualization")
            return

        # Define tabs for categorized visualizations
        tabs = st.tabs([
            "Portfolio Overview", "Demographic Insights", "Investment Behavior",
            "Cluster Profiles", "Risk and Return Analysis", "Correlation Studies"
        ])

        with tabs[0]:  # Portfolio Overview
            st.markdown("### Portfolio Summary")
            col1, col2 = st.columns(2)
            with col1:
                # 1. Customer Distribution by Cluster (Pie Chart)
                cluster_counts = self.df_clustered['kmeans_cluster'].value_counts().sort_index()
                fig1 = px.pie(
                    names=[self.get_cluster_description(i)['name'] for i in cluster_counts.index],
                    values=cluster_counts.values,
                    title="Customer Distribution by Cluster",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig1.update_layout(
                    height=450,
                    title_font_size=20,
                    legend_title_text="Clusters",
                    annotations=[dict(text=f'Avg: {cluster_counts.mean():.0f}', x=0.5, y=0.5, font_size=14, showarrow=False)]
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # 2. Average ROI by Cluster (Bar Chart)
                roi_by_cluster = self.df_clustered.groupby('kmeans_cluster')['roi'].mean().sort_index()
                fig2 = px.bar(
                    x=[self.get_cluster_description(i)['name'] for i in roi_by_cluster.index],
                    y=roi_by_cluster.values,
                    title="Average ROI by Cluster",
                    labels={'x': 'Cluster', 'y': 'ROI (%)'},
                    color_discrete_sequence=['#003087']
                )
                fig2.add_hline(y=roi_by_cluster.mean(), line_dash="dash", annotation_text=f"Avg ROI: {roi_by_cluster.mean():.1f}%", annotation_position="top right")
                fig2.update_layout(height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[1]:  # Demographic Insights
            st.markdown("### Customer Demographics")
            col1, col2 = st.columns(2)
            with col1:
                # 3. Age Distribution (Histogram)
                fig3 = px.histogram(
                    self.df_clustered,
                    x='age',
                    title="Age Distribution Across Customers",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig3.add_vline(x=self.df_clustered['age'].mean(), line_dash="dash", annotation_text=f"Avg Age: {self.df_clustered['age'].mean():.0f}", annotation_position="top right")
                fig3.update_layout(height=450, title_font_size=20, bargap=0.1)
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                # 4. Net Worth vs Age (Scatter)
                fig4 = px.scatter(
                    self.df_clustered,
                    x='age',
                    y='net_worth',
                    color='kmeans_cluster',
                    title="Net Worth vs Age by Cluster",
                    color_continuous_scale='Viridis',
                    size_max=15,
                    labels={'net_worth': 'Net Worth (₹)', 'age': 'Age (Years)'}
                )
                fig4.add_hline(y=self.df_clustered['net_worth'].mean(), line_dash="dash", annotation_text=f"Avg Net Worth: {self.df_clustered['net_worth'].mean()/100000:.1f}L", annotation_position="top right")
                fig4.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig4, use_container_width=True)

            with col2:
                # 5. Tenure Distribution (Bar Chart)
                tenure_counts = self.df_clustered['tenure'].value_counts().sort_index()
                fig5 = px.bar(
                    x=tenure_counts.index,
                    y=tenure_counts.values,
                    title="Distribution of Investing Experience (Years)",
                    color_discrete_sequence=['#003087'],
                    text=tenure_counts.values
                )
                fig5.update_traces(textposition='auto')
                fig5.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig5, use_container_width=True)

        with tabs[2]:  # Investment Behavior
            st.markdown("### Investment Patterns")
            col1, col2 = st.columns(2)
            with col1:
                # 6. Investment Type Popularity (Bar Chart)
                inv_type_counts = self.df_clustered['inv_type'].value_counts().sort_values(ascending=False)
                fig6 = px.bar(
                    x=inv_type_counts.index,
                    y=inv_type_counts.values,
                    title="Popularity of Investment Types",
                    color_discrete_sequence=['#003087'],
                    text=inv_type_counts.values
                )
                fig6.update_traces(textposition='auto')
                fig6.update_layout(height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig6, use_container_width=True)

            with col2:
                # 7. Frequency vs ROI (Scatter)
                fig7 = px.scatter(
                    self.df_clustered,
                    x='freq',
                    y='roi',
                    color='kmeans_cluster',
                    title="Investment Frequency vs ROI",
                    color_continuous_scale='Viridis',
                    size='risk_score',
                    size_max=20,
                    labels={'freq': 'Frequency (per year)', 'roi': 'ROI (%)'}
                )
                fig7.add_hline(y=self.df_clustered['roi'].mean(), line_dash="dash", annotation_text=f"Avg ROI: {self.df_clustered['roi'].mean():.1f}%", annotation_position="top right")
                fig7.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig7, use_container_width=True)

            with col2:
                # 8. Duration vs ROI (Line Chart)
                duration_roi = self.df_clustered.groupby('duration')['roi'].mean().reset_index()
                fig8 = px.line(
                    duration_roi,
                    x='duration',
                    y='roi',
                    title="Average ROI by Investment Duration",
                    labels={'duration': 'Duration (months)', 'roi': 'ROI (%)'},
                    markers=True,
                    color_discrete_sequence=['#1f77b4']
                )
                fig8.add_hline(y=duration_roi['roi'].mean(), line_dash="dash", annotation_text=f"Avg ROI: {duration_roi['roi'].mean():.1f}%", annotation_position="top right")
                fig8.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig8, use_container_width=True)

        with tabs[3]:  # Cluster Profiles
            st.markdown("### Cluster-Specific Insights")
            col1, col2 = st.columns(2)
            with col1:
                # 9. Goal Distribution by Cluster (Stacked Bar)
                goal_pivot = self.df_clustered.groupby(['kmeans_cluster', 'goal']).size().unstack(fill_value=0)
                fig9 = go.Figure()
                for goal in goal_pivot.columns:
                    fig9.add_trace(go.Bar(
                        x=[self.get_cluster_description(i)['name'] for i in goal_pivot.index],
                        y=goal_pivot[goal],
                        name=goal.capitalize(),
                        marker_color=px.colors.qualitative.Plotly[goal_pivot.columns.get_loc(goal) % len(px.colors.qualitative.Plotly)]
                    ))
                fig9.update_layout(barmode='stack', title="Goal Distribution by Cluster", height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig9, use_container_width=True)

            with col2:
                # 10. Goal vs Net Worth (Box Plot)
                fig10 = px.box(
                    self.df_clustered,
                    x='goal',
                    y='net_worth',
                    title="Net Worth Distribution by Financial Goal",
                    color_discrete_sequence=['#003087']
                )
                fig10.add_hline(y=self.df_clustered['net_worth'].mean(), line_dash="dash", annotation_text=f"Avg Net Worth: {self.df_clustered['net_worth'].mean()/100000:.1f}L", annotation_position="top right")
                fig10.update_layout(height=450, title_font_size=20, xaxis_tickangle=45)
                st.plotly_chart(fig10, use_container_width=True)

        with tabs[4]:  # Risk and Return Analysis
            st.markdown("### Risk and Return Metrics")
            col1, col2 = st.columns(2)
            with col1:
                # 11. Risk Score Distribution (Violin)
                fig11 = px.violin(
                    self.df_clustered,
                    y='risk_score',
                    box=True,
                    points="all",
                    color='kmeans_cluster',
                    title="Risk Score Distribution by Cluster",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig11.add_hline(y=self.df_clustered['risk_score'].mean(), line_dash="dash", annotation_text=f"Avg Risk: {self.df_clustered['risk_score'].mean():.1f}", annotation_position="top right")
                fig11.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig11, use_container_width=True)

            with col2:
                # 12. Risk Score Trend (Line Chart)
                risk_by_age = self.df_clustered.groupby('age')['risk_score'].mean().reset_index()
                fig12 = px.line(
                    risk_by_age,
                    x='age',
                    y='risk_score',
                    title="Average Risk Score by Age",
                    labels={'age': 'Age (Years)', 'risk_score': 'Risk Score'},
                    markers=True,
                    color_discrete_sequence=['#1f77b4']
                )
                fig12.add_hline(y=risk_by_age['risk_score'].mean(), line_dash="dash", annotation_text=f"Avg Risk: {risk_by_age['risk_score'].mean():.1f}", annotation_position="top right")
                fig12.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig12, use_container_width=True)

            with col2:
                # 13. Saving Distribution (Histogram)
                fig13 = px.histogram(
                    self.df_clustered,
                    x='saving',
                    title="Distribution of Monthly Savings",
                    nbins=20,
                    color_discrete_sequence=['#003087']
                )
                fig13.add_vline(x=self.df_clustered['saving'].mean(), line_dash="dash", annotation_text=f"Avg Saving: {self.df_clustered['saving'].mean():.0f}₹", annotation_position="top right")
                fig13.update_layout(height=450, title_font_size=20, bargap=0.1)
                st.plotly_chart(fig13, use_container_width=True)

        with tabs[5]:  # Correlation Studies
            st.markdown("### Data Correlations")
            col1, col2 = st.columns(2)
            with col1:
                # 14. Feature Correlation Heatmap
                numerical_cols = ['age', 'saving', 'net_worth', 'risk_score', 'tenure', 'duration', 'freq', 'roi']
                corr_matrix = self.df_clustered[numerical_cols].corr()
                fig14 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    showscale=True
                ))
                fig14.update_layout(title="Correlation Between Numerical Features", height=450, title_font_size=20)
                st.plotly_chart(fig14, use_container_width=True)

            with col2:
                # 15. Saving vs Net Worth (Scatter)
                fig15 = px.scatter(
                    self.df_clustered,
                    x='saving',
                    y='net_worth',
                    color='kmeans_cluster',
                    title="Saving vs Net Worth by Cluster",
                    color_continuous_scale='Viridis',
                    size='roi',
                    size_max=20,
                    labels={'saving': 'Monthly Saving (₹)', 'net_worth': 'Net Worth (₹)'}
                )
                fig15.add_hline(y=self.df_clustered['net_worth'].mean(), line_dash="dash", annotation_text=f"Avg Net Worth: {self.df_clustered['net_worth'].mean()/100000:.1f}L", annotation_position="top right")
                fig15.update_layout(height=450, title_font_size=20)
                st.plotly_chart(fig15, use_container_width=True)
    
    def model_insights_page(self):
        """Model insights page with IBM branding and optimal cluster analysis"""
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=200)
        
        st.markdown('<h1 class="main-header">🤖 Model Insights</h1>', unsafe_allow_html=True)
        
        if not self.models_loaded:
            st.warning("⚠️ Models not loaded. Please run the training pipeline first.")
            return
        
        st.markdown("## 🔧 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### KMeans Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Optimal Clusters:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if 'kmeans_silhouette' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.metadata["kmeans_silhouette"]:.3f}</div>', unsafe_allow_html=True)
            if 'kmeans_inertia' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Inertia:</strong> {self.metadata["kmeans_inertia"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><strong>Algorithm:</strong> K-Means Clustering</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### GMM Model")
            if 'best_k' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Components:</strong> {self.metadata["best_k"]}</div>', unsafe_allow_html=True)
            if 'gmm_silhouette' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>Silhouette Score:</strong> {self.metadata["gmm_silhouette"]:.3f}</div>', unsafe_allow_html=True)
            if 'gmm_bic' in self.metadata:
                st.markdown(f'<div class="metric-card"><strong>BIC Score:</strong> {self.metadata["gmm_bic"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><strong>Algorithm:</strong> Gaussian Mixture Model</div>', unsafe_allow_html=True)
        
        # Optimal Cluster Analysis
        st.markdown("## 🔍 Optimal Cluster Analysis")
        st.image("outputs/optimal_clusters_analysis.png", caption="Optimal Clusters Analysis",)
        st.markdown("""
                The elbow curve, generated by train_models.py, shows the inertia (within-cluster sum of squares) 
                for different numbers of clusters. The optimal number of clusters is typically where the inertia 
                starts to decrease more slowly, forming an "elbow" shape, indicating a balance between cluster 
                compactness and complexity.
                """)
        st.markdown("""The silhouette score plot, generated by train_models.py, measures how similar an object is to 
                its own cluster compared to other clusters. Higher scores indicate better-defined clusters. 
                The optimal number of clusters often corresponds to a peak in the silhouette score, 
                balancing cohesion and separation.""")
        
        # Feature importance visualization
        if 'feature_names' in self.metadata:
            st.markdown("## 📊 Feature Analysis")
            features_df = pd.DataFrame({
                'Feature': self.metadata['feature_names'],
                'Type': ['Numerical' if 'encoded' not in f else 'Categorical' for f in self.metadata['feature_names']]
            })
            fig = px.bar(
                features_df,
                x='Feature',
                y=[1] * len(features_df),
                color='Type',
                title='Features Used in Clustering',
                color_discrete_map={'Numerical': '#003087', 'Categorical': '#1f77b4'}
            )
            fig.update_layout(showlegend=True, height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
The feature analysis highlights the 10 key features used in this investment customer segmentation project: age, saving, net_worth, risk_score, tenure, duration, freq, roi, goal_encoded, and inv_type_encoded. Each feature plays a critical role in understanding customer behavior and investment preferences.

- **Age**: Reflects the life stage and risk appetite, influencing investment horizons and product preferences.
- **Saving**: Indicates monthly financial commitment, a key determinant of investment capacity and strategy.
- **Net Worth**: Represents overall wealth, guiding the allocation towards wealth preservation or growth.
- **Risk Score**: Quantifies risk tolerance, essential for tailoring investment recommendations.
- **Tenure**: Measures investing experience, affecting the complexity of investment choices.
- **Duration**: Captures investment time horizon, critical for aligning with financial goals.
- **Freq**: Tracks investment frequency, indicating engagement and strategy consistency.
- **ROI**: Expected return on investment, vital for assessing performance and cluster profitability.
- **Goal Encoded**: Encoded financial goals (e.g., retirement, wealth) drive personalized investment planning.
- **Inv Type Encoded**: Encoded investment types (e.g., stocks, mutual funds) shape portfolio diversification.

These features collectively enable the model to segment customers effectively, providing actionable insights for targeted financial strategies.
""")
        
        # Model files
        st.markdown("## 📁 Model Artifacts")
        model_files = [
            ("Scaler", "models/scaler.pkl"),
            ("KMeans Model", "models/kmeans_model.pkl"), 
            ("GMM Model", "models/gmm_model.pkl"),
            ("Label Encoders", "models/label_encoders.pkl"),
            ("Metadata", "models/model_metadata.json")
        ]
        
        for name, file_path in model_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024
                st.markdown(f'<div class="metric-card">✅ {name}: {file_size:.1f} KB</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card">❌ {name}: Missing</div>', unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=150)
        
        st.sidebar.title("🧭 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Customer Segmentation", "Analytics Dashboard", "Model Insights"],
            label_visibility="collapsed"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏢 IBM Investment Analytics")
        st.sidebar.markdown("*Powered by Machine Learning*")
        
        if page == "Customer Segmentation":
            self.customer_segmentation_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()
        elif page == "Model Insights":
            self.model_insights_page()
        
        st.sidebar.markdown("---")
       


def main():
    """Main application entry point"""
    app = InvestmentSegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
