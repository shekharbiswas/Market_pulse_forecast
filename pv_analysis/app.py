import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and prepare data
@st.cache
def load_data():
    leads = pd.read_csv("leads.csv")
    funnel = pd.read_csv("sales_funnel.csv")

    # Convert dates
    leads['LEAD_CREATED_DATE'] = pd.to_datetime(leads['LEAD_CREATED_DATE'], errors='coerce')
    funnel['CASE_CLOSED_SUCCESSFUL_DATE'] = pd.to_datetime(funnel['CASE_CLOSED_SUCCESSFUL_DATE'], errors='coerce')

    # Merge
    df = pd.merge(funnel, leads, on="LEAD_ID", how="left")

    # Filter only funnel stages of interest
    steps = ['Sales Call 1', 'Sales Call 2', 'PV System Sold']
    df = df[df['SALES_FUNNEL_STEPS'].isin(steps)]

    # Compute duration
    df['days_to_stage_closed'] = (df['CASE_CLOSED_SUCCESSFUL_DATE'] - df['LEAD_CREATED_DATE']).dt.days

    return df, steps

df, steps = load_data()

# Streamlit - percentile selection dropdown
percentile = st.selectbox("Select Percentile for Analysis", ["25th", "50th", "75th", "90th"])

# Setup plot
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "Lead → Sales Call 1 Closed",
        "Lead → Sales Call 2 Closed",
        "Lead → PV System Closed"
    ]
)

colors = ['lightblue', 'lightgreen', 'salmon']
short_labels = ['Call 1 Closed', 'Call 2 Closed', 'PV Closed']

# Mapping percentile selection to actual values
percentile_dict = {
    "25th": 0.25,
    "50th": 0.50,
    "75th": 0.75,
    "90th": 0.90
}

# Add plots and lines
for i, (step, color, label) in enumerate(zip(steps, colors, short_labels)):
    step_data = df[df['SALES_FUNNEL_STEPS'] == step]['days_to_stage_closed'].dropna()
    selected_percentile = step_data.quantile(percentile_dict[percentile])

    # Histogram
    fig.add_trace(go.Histogram(
        x=step_data,
        nbinsx=30,
        marker_color=color,
        name=label,
        showlegend=True,
        legendgroup=label
    ), row=1, col=i+1)

    # Percentile line
    fig.add_shape(type="line",
        x0=selected_percentile, x1=selected_percentile, y0=0, y1=1,
        xref=f"x{i+1}", yref="paper",
        line=dict(color="blue", dash="dot")
    )

    fig.add_annotation(
        x=selected_percentile, y=1.02,
        text=f"{percentile} Percentile: {int(selected_percentile)}d",
        showarrow=False,
        xref=f"x{i+1}", yref="paper",
        font=dict(color="blue")
    )

# Final layout
fig.update_layout(
    title="Lead Inactivity Safe Window",
    height=500,
    width=1300,
    bargap=0.1,
    legend_title="Safe Window Percentile"
)

fig.update_xaxes(title_text="Days")
fig.update_yaxes(title_text="Lead Count")

# Display the plot in Streamlit
st.plotly_chart(fig)
