import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from hydralit_components import HyLoader, Loaders
import time

# Inject CSS for permanent dark theme
def inject_dark_mode_css():
    bg_color = "#121212"
    text_color = "#FFFFFF"

    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        .main-header {{
            text-align: center;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            animation: glow 2s ease-in-out infinite alternate;
        }}
        @keyframes glow {{
            from {{ filter: drop-shadow(0 0 5px rgba(255, 107, 107, 0.3)); }}
            to {{ filter: drop-shadow(0 0 20px rgba(78, 205, 196, 0.6)); }}
        }}
        .subtitle {{
            text-align: center;
            font-size: 1.2rem;
            color: #CCCCCC;
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .stButton > button {{
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
    </style>
    """, unsafe_allow_html=True)

inject_dark_mode_css()

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Calculator",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Animated loading function
def show_loading_animation():
    with HyLoader(
        "",              # message
        Loaders.standard_loaders,  # loader palette
        index=3                    # choose a loader by index
    ):
        time.sleep(2.5)
    time.sleep(0.5)

if 'loaded' not in st.session_state:
    show_loading_animation()
    st.session_state.loaded = True

st.markdown('<h1 class="main-header">⚡ Energy Consumption Calculator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Calculate your household\'s total energy consumption based on your living situation and appliances.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Personal Information")
    name = st.text_input("Name", placeholder="Enter your name")
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    city = st.text_input("City", placeholder="Enter your city")
    area = st.text_input("Area", placeholder="Enter your area")

with col2:
    st.subheader("🏠 Housing Information")
    housing_type = st.selectbox("Do you own a Flat or Tenement?", ["Flat", "Tenement"])
    bhk = int(st.selectbox("How many BHK is it?", [1, 2, 3], index=0))
    st.subheader("🔌 Appliances")
    ac = st.number_input("Number of Air Conditioners", min_value=0, value=0, step=1)
    fridge = st.number_input("Number of Refrigerators", min_value=0, value=1, step=1)
    wm = st.number_input("Number of Washing Machines", min_value=0, value=0, step=1)

if st.button("Calculate Energy Consumption", type="primary"):
    if name and city and area:
        base_energy = {1: 2.4, 2: 3.6, 3: 4.8}
        total_energy = base_energy[bhk] + (ac * 3) + (fridge * 4) + (wm * 3)

        st.success("✅ Calculation Complete!")
        st.subheader("📊 Results")

        with st.expander("👤 User Information"):
            st.write(f"**Name:** {name}")
            st.write(f"**Age:** {age} years")
            st.write(f"**Location:** {area}, {city}")
            st.write(f"**Housing:** {bhk} BHK {housing_type}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Energy (BHK)", f"{base_energy[bhk]} kWh")
            st.metric("AC Energy", f"{ac * 3} kWh")
        with col2:
            st.metric("Fridge Energy", f"{fridge * 4} kWh")
            st.metric("Washing Machine Energy", f"{wm * 3} kWh")

        st.metric("🔋 Total Energy Consumption", f"{total_energy} kWh", delta=f"{total_energy - base_energy[bhk]} kWh from appliances")

        st.subheader("💡 Energy Insights")
        if total_energy > 10:
            st.warning("⚠️ High energy consumption detected. Consider energy-efficient appliances.")
        elif total_energy > 5:
            st.info("ℹ️ Moderate energy consumption. You're doing well!")
        else:
            st.success("🌱 Low energy consumption. Great job on being energy efficient!")

        st.subheader("📊 Energy Consumption Visualizations")
        breakdown_data = {
            "Source": ["Base (BHK)", "Air Conditioners", "Refrigerators", "Washing Machines"],
            "Energy (kWh)": [base_energy[bhk], ac * 3, fridge * 4, wm * 3],
            "Color": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        }

        filtered_data = {
            k: [v for v, e in zip(breakdown_data[k], breakdown_data["Energy (kWh)"]) if e > 0]
            for k in breakdown_data
        }

        plotly_template = "plotly_dark"

        if filtered_data["Source"]:
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📈 Gauge Chart"])

            with tab1:
                fig_bar = px.bar(
                    x=filtered_data["Source"],
                    y=filtered_data["Energy (kWh)"],
                    color=filtered_data["Color"],
                    title="Energy Consumption by Source",
                    labels={"x": "Energy Source", "y": "Energy Consumption (kWh)"},
                    color_discrete_sequence=filtered_data["Color"]
                )
                fig_bar.update_layout(template=plotly_template, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab2:
                fig_pie = px.pie(
                    values=filtered_data["Energy (kWh)"],
                    names=filtered_data["Source"],
                    title="Energy Distribution",
                    color_discrete_sequence=filtered_data["Color"]
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(template=plotly_template)
                st.plotly_chart(fig_pie, use_container_width=True)

            with tab3:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=total_energy,
                    delta={'reference': 5},
                    title={'text': "Total Energy Consumption (kWh)"},
                    gauge={
                        'axis': {'range': [None, 20]},
                        'bar': {'color': "#FF6B6B"},
                        'steps': [
                            {'range': [0, 5], 'color': "#C8E6C9"},
                            {'range': [5, 10], 'color': "#FFE0B2"},
                            {'range': [10, 20], 'color': "#FFCDD2"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 15}
                    }
                ))
                fig_gauge.update_layout(template=plotly_template)
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("⚖️ Energy Comparison")
        comparison_data = pd.DataFrame({
            'Category': ['Your Consumption', 'Average 1 BHK', 'Average 2 BHK', 'Average 3 BHK'],
            'Energy (kWh)': [total_energy, 5.4, 7.6, 9.8],
            'Type': ['Your Home', 'Benchmark', 'Benchmark', 'Benchmark']
        })
        fig_comparison = px.bar(
            comparison_data,
            x='Category',
            y='Energy (kWh)',
            color='Type',
            title="How does your energy consumption compare?",
            color_discrete_map={'Your Home': '#FF6B6B', 'Benchmark': '#4ECDC4'}
        )
        fig_comparison.update_layout(template=plotly_template)
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.subheader("💡 Energy Efficiency Tips")
        tips = [
            "🌡️ Set your AC to 24°C or higher to save energy",
            "🔌 Unplug appliances when not in use",
            "💡 Switch to LED bulbs for better efficiency",
            "🚿 Use cold water for washing clothes when possible",
            "🏠 Improve insulation to reduce heating/cooling needs",
            "⏰ Use programmable thermostats for better control"
        ]
        for tip in tips:
            st.markdown(f"• {tip}")
    else:
        st.error("❌ Please fill in all required fields (Name, City, Area)")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
