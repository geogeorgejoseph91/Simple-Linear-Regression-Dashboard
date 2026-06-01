# =====================================================================
# MACHINE LEARNING: MODULAR SIMPLE LINEAR REGRESSION DASHBOARD
# =====================================================================
# This file is structured modularly. The code is divided into helper 
# functions, data components, and UI widgets to maintain high readability,
# clean separation of concerns, and ease of understanding for beginners.

# ---------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# ---------------------------------------------------------------------
import streamlit as st        # For layout design and web app components
import pandas as pd           # For working with tables and datasets
import numpy as np            # For math logic (e.g. square roots, linear linspace)
import os                     # For reading file paths from local drive
import plotly.graph_objects as go  # For drawing beautiful interactive charts

# Scikit-learn algorithms and score calculation tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# MODULE A: UTILITY & HELPER FUNCTIONS
# =====================================================================

def is_currency(col_name: str) -> bool:
    """
    Checks if a column name contains keywords related to monetary values.
    Returns True if found, False otherwise.
    """
    col_lower = col_name.lower()
    currency_keywords = ["price", "cost", "value", "val", "amount", "salary", "revenue", "income", "expense", "budget"]
    return any(kw in col_lower for kw in currency_keywords)


def get_currency_symbol(col_name: str) -> str:
    """
    Returns the appropriate currency symbol depending on column name text.
    Defaults to Indian Rupee (₹).
    """
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in ["usd", "dollar", "$"]):
        return "$"
    elif any(kw in col_lower for kw in ["eur", "euro", "€"]):
        return "€"
    elif any(kw in col_lower for kw in ["gbp", "pound", "£"]):
        return "£"
    return "₹"


# =====================================================================
# MODULE B: STYLING & PAGE INITIALIZATION
# =====================================================================

def inject_custom_styles():
    """
    Injects custom CSS styling to modernize the Streamlit page with
    premium dark mode themes, capsule tabs, and hover animations.
    """
    st.markdown("""
        <style>
        /* Main background area styling */
        .main {
            background-color: #0f172a; 
        }

        /* Increase main block container width to use screen efficiently */
        .block-container {
            max-width: 95% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Sidebar container styling */
        
        /* Page Header Font Styling */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            color: #f8fafc;
        }

        /* Streamlit Tabs Navigation Bar Custom Styling (No background) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent !important;
            border: none !important;
            padding: 0px !important;
            margin-bottom: 25px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            padding: 0 20px;
            color: #94a3b8;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35);
            border: none !important;
        }
        
        /* Custom card styling border overrides for native Streamlit containers */
        div[data-testid="stVerticalBlockBorderConstraint"] {
            background-color: rgba(30, 41, 59, 0.25) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Prediction Card box (gradient and borders, no inline style) */
        .prediction-highlight-card {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%) !important;
            border: 1px solid rgba(6, 182, 212, 0.3) !important;
            border-radius: 14px !important;
            padding: 20px !important;
            text-align: center !important;
            margin-top: 15px !important;
            margin-bottom: 0px !important;
        }
        .prediction-highlight-card p {
            margin: 0 !important;
            padding: 0 !important;
        }
        .prediction-highlight-label {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            margin-bottom: 5px !important;
            text-transform: uppercase !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
        }
        .prediction-highlight-value {
            color: #06b6d4 !important;
            font-family: 'Outfit', sans-serif !important;
            margin: 0 !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }
        
        /* Metric card styling: background #f2f2f2, padding 10px, dark text colors */
        div[data-testid="stMetric"] {
            background-color: #f2f2f2 !important;
            padding: 10px !important;
            border-radius: 10px !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.05em !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
        }
        
        /* Add top margin to main page notification banner (active config) */
        .main div[data-testid="stNotification"],
        .main div[data-testid="stAlert"] {
            margin-top: 30px !important;
        }
        </style>
    """, unsafe_allow_html=True)


# =====================================================================
# MODULE C: DATA COMPONENT (LOADING DATA)
# =====================================================================

def load_dataset() -> pd.DataFrame:
    """
    Loads dataset from either file uploader in the sidebar or defaults
    to the local 'house.csv' file. Returns a Pandas DataFrame.
    """
    # Clean sidebar title
    st.sidebar.markdown("""
        <div style="padding-bottom: 10px;">
            <h2 style="color: #06b6d4; font-family: 'Outfit'; font-size: 1.4rem; margin: 0; font-weight: 800;">📁 Data Source</h2>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    # If a custom file is uploaded:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset loaded successfully!")
            return df
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
            return None
            
    # Default fallback:
    default_csv = "house.csv"
    if os.path.exists(default_csv):
        st.sidebar.info("Using default 'house.csv' dataset.")
        return pd.read_csv(default_csv)
    else:
        st.sidebar.warning("No dataset found. Please upload a CSV file.")
        return None


# =====================================================================
# MODULE D: MODEL CONFIGURATION COMPONENT
# =====================================================================

def get_model_config(df: pd.DataFrame):
    """
    Renders selection widgets in the sidebar to configure the regression
    model inputs. Returns feature column, target column, test split ratio,
    and random state seed.
    """
    # Clean config header
    st.sidebar.markdown("""
        <div style="padding-top: 15px; padding-bottom: 10px; border-top: 1px solid rgba(255, 255, 255, 0.05);">
            <h2 style="color: #06b6d4; font-family: 'Outfit'; font-size: 1.4rem; margin: 0; font-weight: 800;">⚙️ Model Settings</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Extract only numeric columns for simple linear regression
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if len(numeric_cols) < 2:
        st.error("The dataset must contain at least 2 numerical columns for simple linear regression.")
        return None, None, None, None
        
    # Auto-detect default columns
    default_x_idx = numeric_cols.index("HouseSizeSqFt") if "HouseSizeSqFt" in numeric_cols else 0
    default_y_idx = numeric_cols.index("HousePrice") if "HousePrice" in numeric_cols else min(1, len(numeric_cols)-1)
    
    # Safety guards for index ranges
    if default_x_idx >= len(numeric_cols): default_x_idx = 0
    if default_y_idx >= len(numeric_cols): default_y_idx = min(1, len(numeric_cols)-1)
        
    # Column selections
    x_col = st.sidebar.selectbox("Select Feature (X)", numeric_cols, index=default_x_idx)
    y_col = st.sidebar.selectbox("Select Target (y)", numeric_cols, index=default_y_idx)
    
    # Split ratio selector
    test_size = st.sidebar.slider("Test Set Split Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    # Random State seed configs
    use_random_state = st.sidebar.checkbox("Use Random State Seed", value=True)
    random_state = None
    if use_random_state:
        random_state = st.sidebar.number_input("Random State Seed Value", min_value=0, max_value=9999, value=42, step=1)
        
    return x_col, y_col, test_size, random_state


# =====================================================================
# MODULE E: MACHINE LEARNING COMPUTATION ENGINE
# =====================================================================

def train_model(df: pd.DataFrame, x_col: str, y_col: str, test_size: float, random_state: int):
    """
    Performs train-test split, trains a LinearRegression model, and
    calculates performance metrics on the test split.
    Returns: Trained model, training/testing splits, coefficients, and metrics.
    """
    X = df[[x_col]]
    y = df[y_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluation Scores
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    }
    
    return model, X_train, X_test, y_train, y_test, y_test_pred, metrics


# =====================================================================
# MODULE F: UI PRESENTATION COMPONENTS
# =====================================================================

def render_formula_and_params(x_col: str, y_col: str, slope: float, intercept: float):
    """
    Renders the regression formula code block and Slope/Intercept metrics.
    """
    with st.container(border=True):
        st.subheader("📊 Regression Formula")
        st.write("Calculated straight-line equation:")
        st.code(f"{y_col} = {slope:.4f} * {x_col} + {intercept:.4f}", language="python")
        
        st.subheader("⚙️ Parameters")
        param_cols = st.columns(3)
        param_cols[0].metric("Slope (m)", f"{slope:,.4f}", help="The change in target for a 1-unit increase in feature.")
        param_cols[1].metric("Intercept (c)", f"{intercept:,.4f}", help="The starting value when feature is 0.")


def render_predictor(df: pd.DataFrame, model: LinearRegression, x_col: str, y_col: str, symbol: str) -> float:
    """
    Renders the slider interface to make real-time predictions.
    Returns the user-selected value for plotting.
    """
    with st.container(border=True):
        st.subheader("🎯 Real-time Predictor")
        st.write("Adjust the input value to see the model's prediction output:")
        
        # Calculate boundaries
        min_val = float(df[x_col].min())
        max_val = float(df[x_col].max())
        step_val = float((max_val - min_val) / 50.0) if max_val > min_val else 1.0
        
        slider_min = float(min_val * 0.5)
        slider_max = float(max_val * 1.5)
        slider_default = float(df[x_col].mean())
        
        if slider_default < slider_min: slider_default = slider_min
        elif slider_default > slider_max: slider_default = slider_max
        if step_val <= 0: step_val = 0.1
            
        user_input = st.slider(
            f"Adjust {x_col}",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
            step=step_val
        )
        
        # Run prediction
        pred_df = pd.DataFrame({x_col: [user_input]})
        predicted_val = model.predict(pred_df)[0]
        
        # Format and show prediction box using modern st.html (fallback to markdown on old versions)
        formatted_prediction = f"{symbol}{predicted_val:,.2f}" if symbol else f"{predicted_val:,.4f}"
        html_code = f"""
        <div class="prediction-highlight-card">
            <div class="prediction-highlight-label">Predicted {y_col}</div>
            <div class="prediction-highlight-value">{formatted_prediction}</div>
        </div>
        """
        if hasattr(st, "html"):
            st.html(html_code)
        else:
            st.markdown(html_code, unsafe_allow_html=True)
    return user_input, predicted_val


def render_metrics(metrics: dict, symbol: str):
    """
    Renders scorecard boxes for the five request evaluation scores in a 3-column layout.
    """
    with st.container(border=True):
        st.subheader("📈 Performance Metrics")
        st.write("Model evaluation on unseen testing data:")
        
        # Row 1: 3 cards
        row1_cols = st.columns(3)
        row1_cols[0].metric("R² Score (R2)", f"{metrics['r2']:.4f}", help="Coefficient of Determination: Closer to 1 is better.")
        row1_cols[1].metric("RMSE", f"{symbol}{metrics['rmse']:,.2f}" if symbol else f"{metrics['rmse']:,.4f}", help="Root Mean Squared Error: Lower is better.")
        row1_cols[2].metric("MSE", f"{metrics['mse']:,.2e}" if metrics['mse'] > 1e6 else f"{metrics['mse']:,.4f}", help="Mean Squared Error: Heavily penalizes large errors.")
        
        # Row 2: 2 cards in a 3-column layout to align perfectly
        row2_cols = st.columns(3)
        row2_cols[0].metric("MAE", f"{symbol}{metrics['mae']:,.2f}" if symbol else f"{metrics['mae']:,.4f}", help="Mean Absolute Error: Average absolute prediction deviation.")
        row2_cols[1].metric("MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error: Average percent deviation from true values.")


def render_regression_plot(df: pd.DataFrame, X_train, y_train, X_test, y_test, slope: float, intercept: float, x_col: str, y_col: str, user_input: float, predicted_val: float):
    """
    Constructs and renders a highly polished, professional Plotly regression chart.
    """
    with st.container(border=True):
        st.subheader("📈 Regression Plot")
        st.write("Fitted line vs train (indigo) and test (sky blue) data points:")
        
        # Determine clean symbol formatting
        symbol = get_currency_symbol(y_col) if is_currency(y_col) else ""
        y_format = f"{symbol}%{{y:,.2f}}"
        y_line_format = f"{symbol}%{{y:,.2f}}"
        
        fig = go.Figure()
        
        # 1. Training Set Scatter Trace (Indigo with white border)
        fig.add_trace(go.Scatter(
            x=X_train[x_col], y=y_train,
            mode='markers', name='Train Data',
            marker=dict(color='#4f46e5', size=8.5, opacity=0.75, line=dict(color='white', width=1)),
            hovertemplate=f"<b>Train Data Point</b><br>{x_col}: %{{x:,.2f}}<br>{y_col}: {y_format}<extra></extra>"
        ))
        
        # 2. Testing Set Scatter Trace (Sky Blue Diamonds with white border)
        fig.add_trace(go.Scatter(
            x=X_test[x_col], y=y_test,
            mode='markers', name='Test Data',
            marker=dict(color='#0ea5e9', size=10, symbol='diamond', line=dict(color='white', width=1)),
            hovertemplate=f"<b>Test Data Point</b><br>{x_col}: %{{x:,.2f}}<br>{y_col}: {y_format}<extra></extra>"
        ))
        
        # 3. Regression Line Trace (Rose red, bounded to actual range of feature values)
        line_x = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        line_y = slope * line_x + intercept
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y,
            mode='lines', name='Regression Line',
            line=dict(color='#f43f5e', width=3),
            hovertemplate=f"<b>Fitted Regression Line</b><br>{x_col}: %{{x:,.2f}}<br>Predicted {y_col}: {y_line_format}<extra></extra>"
        ))
        
        # 4. User input prediction tracker (Violet with dual-ring halo)
        fig.add_trace(go.Scatter(
            x=[user_input], y=[predicted_val],
            mode='markers', name='User Prediction',
            marker=dict(color='#8b5cf6', size=14, line=dict(color='white', width=2)),
            hovertemplate=f"<b>User Prediction</b><br>Adjusted {x_col}: %{{x:,.2f}}<br>Predicted {y_col}: {y_line_format}<extra></extra>"
        ))
        
        # Professional layout configuration
        y_axis_config = dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            zerolinecolor='#cbd5e1',
            tickformat=",",
            tickfont=dict(size=11, family='Inter')
        )
        if symbol:
            y_axis_config['tickprefix'] = symbol

        fig.update_layout(
            xaxis=dict(
                title=dict(text=x_col, font=dict(size=12, family='Inter')),
                showgrid=True,
                gridcolor='#f1f5f9',
                zerolinecolor='#cbd5e1',
                tickformat=",",
                tickfont=dict(size=11, family='Inter')
            ),
            yaxis=dict(
                title=dict(text=y_col, font=dict(size=12, family='Inter')),
                **y_axis_config
            ),
            margin=dict(l=50, r=20, t=30, b=70),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#1e293b', family='Inter'),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_residuals(y_test, y_test_pred, symbol: str):
    """
    Renders actual vs predicted values table with residual calculations.
    """
    with st.container(border=True):
        st.subheader("📝 Predictions & Residuals")
        st.write("Compare actual target values with model predictions:")
        
        test_results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_test_pred,
            "Difference (Residual)": y_test - y_test_pred
        })
        
        if symbol:
            formatted_style = test_results.style.format({
                "Actual": lambda x: f"{symbol}{x:,.2f}",
                "Predicted": lambda x: f"{symbol}{x:,.2f}",
                "Difference (Residual)": lambda x: f"{symbol}{x:,.2f}"
            })
        else:
            formatted_style = test_results.style.format({
                "Actual": "{:,.4f}",
                "Predicted": "{:,.4f}",
                "Difference (Residual)": "{:,.4f}"
            })
        st.dataframe(formatted_style, use_container_width=True, height=350)


def render_raw_data(df: pd.DataFrame):
    """
    Renders the raw DataFrame preview at the bottom.
    """
    with st.container(border=True):
        st.subheader("📋 Raw Dataset Preview")
        st.write("Inspect the loaded dataset records:")
        st.dataframe(df, use_container_width=True, height=350)


def render_step_by_step_code(x_col: str, y_col: str, test_size: float, random_state: int):
    """
    Generates and renders dynamic, copyable step-by-step python regression code.
    """
    st.subheader("📖 Step-by-Step Python Implementation Code")
    st.write("Copy and run this exact model code locally to reproduce this regression analysis:")
    
    dynamic_code = f"""# =====================================================================
# STEP-BY-STEP SIMPLE LINEAR REGRESSION RUNNABLE CODE
# =====================================================================
# This script loads data, splits it into training and testing sets, 
# fits a linear regression model, prints performance metrics, and plots the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load your dataset (.csv file)
# Replace this filename with your local path if running on your machine
df = pd.read_csv("your_uploaded_dataset.csv")

# Step 2: Define the Feature (X) and Target (y)
# X must be a DataFrame (2D shape), y must be a Series (1D shape)
X = df[["{x_col}"]]
y = df["{y_col}"]

# Step 3: Train-Test Split (using selected options)
# Split the dataset into:
# - X_train, y_train (for training/learning)
# - X_test, y_test (for validating accuracy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size={test_size}, 
    random_state={random_state}
)

# Step 4: Initialize and Train Simple Linear Regression Model
# This fits a line y = mx + c that minimizes squared prediction errors
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Extract Model Coefficients (Slope and Intercept)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression Equation: {y_col} = {{slope:.4f}} * {x_col} + {{intercept:.4f}}")
print(f"Slope (Coefficient / m): {{slope}}")
print(f"Intercept (Bias / c): {{intercept}}")

# Step 6: Make Predictions on Test Set
# Use the trained coefficients to predict y values for the test features
y_pred = model.predict(X_test)

# Step 7: Calculate Evaluation Metrics
# These scores measure how far our predictions are from actual values
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Step 8: Print Scores
print("\\nModel Evaluation Scores:")
print(f"R² Score (Accuracy): {{r2:.4f}}")
print(f"RMSE (Root Mean Sq. Error): {{rmse:.4f}}")
print(f"MSE (Mean Sq. Error): {{mse:.4f}}")
print(f"MAE (Mean Absolute Error): {{mae:.4f}}")
print(f"MAPE (Percentage Error): {{mape:.2f}}%")

# Step 9: Plot Actual Data vs Regression Line using Matplotlib
plt.figure(figsize=(10, 6))

# Plot training points (Blue)
plt.scatter(X_train["{x_col}"], y_train, color="#3b82f6", label="Train Data", alpha=0.7, edgecolors="white")

# Plot test points (Cyan diamonds)
plt.scatter(X_test["{x_col}"], y_test, color="#06b6d4", label="Test Data", marker="D", alpha=0.8, edgecolors="white")

# Draw fitted regression line (Red)
x_line = np.linspace(X["{x_col}"].min() * 0.8, X["{x_col}"].max() * 1.2, 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color="#ef4444", linewidth=3, label="Regression Line")

plt.title("Simple Linear Regression: {x_col} vs {y_col}", fontsize=14, fontweight="bold")
plt.xlabel("{x_col}", fontsize=12)
plt.ylabel("{y_col}", fontsize=12)
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
"""
    st.code(dynamic_code, language="python")
    
    # Provide option to download the code as a .py file
    st.download_button(
        label="📥 Download Python Code File",
        data=dynamic_code,
        file_name="simple_linear_regression.py",
        mime="text/x-python"
    )


# =====================================================================
# MAIN FLOW ORCHESTRATOR
# =====================================================================

def main():
    # 1. Page styles & setup
    st.set_page_config(
        page_title="Simple Linear Regression Dashboard",
        page_icon="📈",
        layout="wide"
    )
    inject_custom_styles()
    
    st.title("📈 Simple Linear Regression Dashboard")
    
    # 2. Loading the dataset
    df = load_dataset()
    
    if df is not None:
        total_cols = len(df.columns)
        
        # Validate that dataset has exactly 2 columns
        if total_cols != 2:
            st.error("### ❌ Simple Regression Requires 2 Columns")
            st.write(
                f"Simple Linear Regression requires a dataset with **exactly 2 columns** "
                f"(1 independent feature and 1 dependent target). The uploaded dataset contains **{total_cols}** columns."
            )
            st.info("💡 **Requirement:** Please upload a CSV file with exactly 2 columns.")
            return

        # Validate that both columns contain continuous numerical data
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 2:
            st.error("### ❌ Non-Continuous Columns Detected")
            st.write(
                "Simple Linear Regression requires both columns to contain **continuous numerical data**. "
                f"The loaded dataset contains only **{len(numeric_cols)}** numerical column(s)."
            )
            st.info("💡 **Requirement:** Please ensure both columns contain continuous numeric decimal or integer values.")
            return

        # 3. Model configurations (Sidebar controls)
        x_col, y_col, test_size, random_state = get_model_config(df)
        
        # Guard clause in case column selections failed
        if x_col is None:
            return
            
        # Ensure independent and dependent columns are distinct
        if x_col == y_col:
            st.error("### ❌ Invalid Column Selection")
            st.write(
                f"Feature (X) and Target (y) must be distinct columns. "
                f"You selected `{x_col}` for both the independent and dependent variables."
            )
            st.info("💡 **Recommendation:** Please select different columns for Feature (X) and Target (y) using the sidebar controls.")
            return
            
        # 4. Train regression model and fetch predictions/metrics
        model, X_train, X_test, y_train, y_test, y_test_pred, metrics = train_model(
            df, x_col, y_col, test_size, random_state
        )
        
        slope = model.coef_[0]
        intercept = model.intercept_
        symbol = get_currency_symbol(y_col) if is_currency(y_col) else ""
        
        st.markdown(
            "<div style='margin-top:40px'></div>",
            unsafe_allow_html=True
        )

        # Display the active configuration settings at the top
        st.info(f"🧬 **Active Configuration:** Feature: `{x_col}` | Target: `{y_col}` | Train Split: `{100*(1-test_size):.0f}%` / Test Split: `{100*test_size:.0f}%` | Random State: `{random_state if random_state is not None else 'None (Dynamic Split)'}`")
        
        # 5. Render Navigation Tabs
        tab_dashboard, tab_code = st.tabs(["📊 Interactive Dashboard", "📖 Step-by-Step Python Code"])
        
        with tab_dashboard:

            # prediction
            user_input, predicted_val = render_predictor(df, model, x_col, y_col, symbol)

            # Regression Formula
            render_formula_and_params(x_col, y_col, slope, intercept)

            # Model Parameters
            render_metrics(metrics, symbol)
            
            render_regression_plot(df, X_train, y_train, X_test, y_test, slope, intercept, x_col, y_col, user_input, predicted_val)
            render_residuals(y_test, y_test_pred, symbol)
            render_raw_data(df)
            
        with tab_code:
            render_step_by_step_code(x_col, y_col, test_size, random_state)

if __name__ == "__main__":
    main()
