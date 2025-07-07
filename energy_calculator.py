import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Calculator",
    page_icon="⚡",
    layout="centered"
)

# Title and description
st.title("⚡ Energy Consumption Calculator")
st.markdown("Calculate your household's total energy consumption based on your living situation and appliances.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Personal Information")
    
    # Name input
    name = st.text_input("Name", placeholder="Enter your name")
    
    # Age input with validation
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    
    # City input
    city = st.text_input("City", placeholder="Enter your city")
    
    # Area input
    area = st.text_input("Area", placeholder="Enter your area")

with col2:
    st.subheader("🏠 Housing Information")
    
    # Housing type
    housing_type = st.selectbox(
        "Do you own a Flat or Tenement?",
        ["Flat", "Tenement"]
    )
    
    # BHK selection
    bhk = st.selectbox(
        "How many BHK is it?",
        [1, 2, 3],
        index=0
    )
    
    st.subheader("🔌 Appliances")
    
    # Appliance inputs
    ac = st.number_input("Number of Air Conditioners", min_value=0, value=0, step=1)
    fridge = st.number_input("Number of Refrigerators", min_value=0, value=1, step=1)
    wm = st.number_input("Number of Washing Machines", min_value=0, value=0, step=1)

# Calculate button
if st.button("Calculate Energy Consumption", type="primary"):
    if name and city and area:
        # Calculate base energy based on BHK
        base_energy = {1: 2.4, 2: 3.6, 3: 4.8}
        total_energy = base_energy[bhk]
        
        # Add appliance energy consumption
        total_energy += (ac * 3) + (fridge * 4) + (wm * 3)
        
        # Display results
        st.success("✅ Calculation Complete!")
        
        # Create results section
        st.subheader("📊 Results")
        
        # Display user info in an expander
        with st.expander("👤 User Information"):
            st.write(f"**Name:** {name}")
            st.write(f"**Age:** {age} years")
            st.write(f"**Location:** {area}, {city}")
            st.write(f"**Housing:** {bhk} BHK {housing_type}")
        
        # Energy breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Base Energy (BHK)", f"{base_energy[bhk]} kWh")
            st.metric("AC Energy", f"{ac * 3} kWh")
        
        with col2:
            st.metric("Fridge Energy", f"{fridge * 4} kWh")
            st.metric("Washing Machine Energy", f"{wm * 3} kWh")
        
        # Total energy consumption
        st.metric(
            "🔋 Total Energy Consumption", 
            f"{total_energy} kWh",
            delta=f"{total_energy - base_energy[bhk]} kWh from appliances"
        )
        
        # Additional insights
        st.subheader("💡 Energy Insights")
        
        
        if total_energy > 10:
            st.warning("⚠️ High energy consumption detected. Consider energy-efficient appliances.")
        elif total_energy > 5:
            st.info("ℹ️ Moderate energy consumption. You're doing well!")
        else:
            st.success("🌱 Low energy consumption. Great job on being energy efficient!")
        
        # Breakdown chart
        if ac > 0 or fridge > 0 or wm > 0:
            st.subheader("📈 Energy Breakdown")
            
            breakdown_data = {
                "Source": ["Base (BHK)", "Air Conditioners", "Refrigerators", "Washing Machines"],
                "Energy (kWh)": [base_energy[bhk], ac * 3, fridge * 4, wm * 3]
            }
            
            # Filter out zero values for cleaner chart
            filtered_data = {
                "Source": [source for source, energy in zip(breakdown_data["Source"], breakdown_data["Energy (kWh)"]) if energy > 0],
                "Energy (kWh)": [energy for energy in breakdown_data["Energy (kWh)"] if energy > 0]
            }
            
            st.bar_chart(dict(zip(filtered_data["Source"], filtered_data["Energy (kWh)"])))
    
    else:
        st.error("❌ Please fill in all required fields (Name, City, Area)")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")