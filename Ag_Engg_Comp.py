import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Set page configuration
st.set_page_config(
    page_title="Dr.Vinoth's Academy",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #388e3c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #43a047;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .formula {
        background-color: #dcdcdc;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #7cb342;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #c8e6c9;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1.5rem 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)





# Data tables for calculations
# Table 3.1: Elevation factors
elevation_data = {
    'Elevation (m)': ['<300', '300', '600', '900', '1200', '1500', '1800', '2100', '2400'],
    'Felev': [1.00, 1.04, 1.08, 1.12, 1.16, 1.20, 1.25, 1.30, 1.3]
}
elevation_df = pd.DataFrame(elevation_data)

# Table 3.2: Light intensity factors
light_data = {
    'Light (k lux)': [43.1, 48.4, 53.8, 59.2, 64.6, 70.0, 75.3, 80.1, 86.1],
    'Flight': [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60]
}
light_df = pd.DataFrame(light_data)

# Table 3.3: Temperature rise factors
temp_data = {
    'Temperature Rise (°C)': [5.6, 5.0, 4.4, 3.9, 3.3, 2.8, 2.2],
    'Ftemp': [0.70, 0.78, 0.88, 1.00, 1.17, 1.40, 1.75]
}
temp_df = pd.DataFrame(temp_data)

# Table 3.4: Pad-to-fan distance factors (first part)
vel_data1 = {
    'Distance (m)': [6.1, 7.6, 9.1, 10.7, 12.2, 13.7, 15.2, 16.8, 18.3],
    'Fvel': [2.24, 2.00, 1.83, 1.69, 1.58, 1.48, 1.41, 1.35, 1.29]
}
vel_df1 = pd.DataFrame(vel_data1)

# Table 3.4: Pad-to-fan distance factors (second part)
vel_data2 = {
    'Distance (m)': [19.8, 21.3, 22.9, 24.4, 25.9, 27.4, 29.0, '>30.5'],
    'Fvel': [1.24, 1.20, 1.16, 1.12, 1.08, 1.05, 1.02, 1.00]
}
vel_df2 = pd.DataFrame(vel_data2)

# Table 3.6: Winter temperature difference factors
winter_data = {
    'Temperature Difference (°C)': [10.0, 9.4, 8.9, 8.3, 7.8, 7.2, 6.7, 6.1, 5.6, 5.0],
    'Fwinter': [0.83, 0.88, 0.94, 1.00, 1.07, 1.15, 1.25, 1.37, 1.50, 1.67]
}
winter_df = pd.DataFrame(winter_data)


# Helper functions for interpolation
def interpolate_value(df, column_name, value_column, lookup_value):
    """Interpolate a value from a dataframe."""
    # Check if we need to handle string values
    contains_strings = False
    for val in df[column_name]:
        if isinstance(val, str):
            contains_strings = True
            break

    if contains_strings:
        # Handle columns with string values like '<300' or '>30.5'
        for i, val in enumerate(df[column_name]):
            # Convert string values to comparable numbers
            numeric_val = val
            if isinstance(val, str):
                if val.startswith('<'):
                    numeric_val = float(val[1:]) - 0.1  # Just below the threshold
                elif val.startswith('>'):
                    numeric_val = float(val[1:]) + 0.1  # Just above the threshold
                else:
                    numeric_val = float(val)
            else:
                numeric_val = float(val)

            # Check if this is where our lookup value falls
            if lookup_value <= numeric_val:
                if i == 0:
                    return df[value_column].iloc[i]
                else:
                    # Get previous value for interpolation
                    prev_val = df[column_name].iloc[i - 1]
                    if isinstance(prev_val, str):
                        if prev_val.startswith('<'):
                            prev_numeric = float(prev_val[1:]) - 0.1
                        elif prev_val.startswith('>'):
                            prev_numeric = float(prev_val[1:]) + 0.1
                        else:
                            prev_numeric = float(prev_val)
                    else:
                        prev_numeric = float(prev_val)

                    # Interpolate between previous and current
                    y0 = df[value_column].iloc[i - 1]
                    y1 = df[value_column].iloc[i]

                    # If exactly at a boundary, return exact value
                    if lookup_value == numeric_val:
                        return y1

                    # Otherwise interpolate
                    return y0 + (y1 - y0) * (lookup_value - prev_numeric) / (numeric_val - prev_numeric)

        # If we get here, the value is beyond the largest in our table
        # Handle the special case for '>30.5' in pad-to-fan distance
        last_val = df[column_name].iloc[-1]
        if isinstance(last_val, str) and last_val.startswith('>'):
            threshold = float(last_val[1:])
            if lookup_value > threshold:
                return df[value_column].iloc[-1]

        # Default to last value if nothing else matches
        return df[value_column].iloc[-1]
    else:
        # Handle purely numeric columns with simpler logic
        if lookup_value <= float(df[column_name].iloc[0]):
            return df[value_column].iloc[0]
        elif lookup_value >= float(df[column_name].iloc[-1]):
            return df[value_column].iloc[-1]
        else:
            for i in range(1, len(df)):
                if lookup_value <= float(df[column_name].iloc[i]):
                    x0 = float(df[column_name].iloc[i - 1])
                    x1 = float(df[column_name].iloc[i])
                    y0 = df[value_column].iloc[i - 1]
                    y1 = df[value_column].iloc[i]
                    return y0 + (y1 - y0) * (lookup_value - x0) / (x1 - x0)
            return df[value_column].iloc[-1]


# Navigation setup at the beginning
st.sidebar.title("Navigation")

page = st.sidebar.radio("Select Calculator",
    ["Introduction", "Summer Cooling System", "Winter Cooling System", "Cereal Grain Analysis",
     "Bulk Density & Porosity", "Grain Moisture Content", "Terminal Velocity",
     "Screen Cleaner Evaluation", "Tray Dryer Evaluation", "Belt Conveyor Evaluation",
     "Bucket Conveyor Evaluation"],
    index=0)

# Introduction page
if page == "Introduction":
    st.title("Agricultural Engineering Companion: Interactive Calculation Platform")

    # Quotes Header
    st.markdown("<h3 style='text-align: center; color: #33691e; margin-bottom: 15px;'>Welcome to Dr.Vinoth's Academy </h3>",
                unsafe_allow_html=True)

    # First quote
    st.markdown("""
    <div style="background: linear-gradient(to right, #f1f8e9, #dcedc8); 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 8px solid #7cb342;
                margin-bottom: 15px;">
        <p style="font-size: 20px; text-align: center; font-style: italic; color: #455a64; line-height: 1.6;">
            "Learning is the only thing the mind never exhausts, never fears, and never regrets."
        </p>
        <p style="text-align: right; color: #546e7a; font-weight: bold;">― Leonardo da Vinci, inventor and polymath</p>
    </div>
    """, unsafe_allow_html=True)

    # Second quote (Fixed block)
    st.markdown("""
    <div style="background: linear-gradient(to right, #f1f8e9, #dcedc8); 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 8px solid #7cb342;
                margin-bottom: 20px;">
        <p style="font-size: 20px; text-align: center; font-style: italic; color: #455a64; line-height: 1.6;">
            "Live as if you were to die tomorrow. Learn as if you were to live forever."
        </p>
        <p style="text-align: right; color: #546e7a; font-weight: bold;">― Mahatma Gandhi</p>
    </div>
    """, unsafe_allow_html=True)

    # About the Creator section
    st.header("About the Creator")

    st.markdown("""
    I am **Dr. VinothKumar**, an Agricultural Engineer specializing in Farm Machinery and Power Engineering, with a Ph.D. that focused on optimizing agricultural equipment design and performance evaluation. My academic journey has been driven by a passion for improving agricultural efficiency through engineering solutions that benefit farmers and the agricultural sector.

    With experience in both academic research and practical field applications, I have developed this comprehensive suite of calculators to bridge the gap between theoretical principles and their real-world implementation. This application was initially designed for B.Sc. Agriculture students studying "Protected Cultivation and Secondary Agriculture," but I have expanded its scope to serve as a valuable resource for agricultural professionals, engineers, researchers, and students across various disciplines.

    My goal is to make complex agricultural engineering calculations accessible and intuitive, empowering the next generation of agricultural professionals with practical tools that enhance their learning and professional capabilities.
    """)

    # Dedication section
    st.header("Dedication")

    st.markdown("""
    This application is dedicated with profound respect and immeasurable gratitude to my mentor and most influential teacher, the late **Dr. J. John Gunasekar**, Professor of Agricultural Engineering.

    Dr. Gunasekar's exceptional guidance transcended traditional academics; he was not merely an instructor of engineering principles, but a visionary who inspired his students to see beyond conventional boundaries. His teaching philosophy emphasized the practical application of theoretical knowledge to solve real-world agricultural challenges.

    His wisdom continues to guide me long after our classroom interactions. The methodical approach to problem-solving, the emphasis on precision in design calculations, and the unwavering commitment to creating technology that serves the farmer's needs—all these values that he instilled remain the foundation of my professional ethos.

    This application embodies his belief that educational tools should be both rigorous in their technical foundation and accessible in their implementation. It is my hope that through this work, a small portion of his tremendous legacy in agricultural engineering education continues to benefit future generations.
    """)

    # Display the image using the Imgur link
    st.image("https://i.imgur.com/uL7HeZs.jpeg", caption="Dr. VinothKumar with Dr. J. John Gunasekar")

    # App information
    st.header("About This Application")

    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #388e3c;">
    This application represents a comprehensive collection of engineering tools designed specifically for Protected Cultivation and Secondary Agriculture. It integrates theoretical principles with practical calculations, providing students and professionals with interactive interfaces to analyze, visualize, and optimize agricultural systems and processes.
    <br><br>
    From environmental control systems in greenhouses to post-harvest processing equipment evaluation, each module has been meticulously developed to facilitate both learning and practical application in agricultural engineering contexts.
    </div>
    """, unsafe_allow_html=True)

    # Available modules
    st.header("Available Modules")

    st.subheader("1. Greenhouse Environment Control")
    st.markdown("""
    - **Summer Cooling System**: Calculate air exchange rates, fan capacity, and pad area for evaporative cooling  
    - **Winter Cooling System**: Determine convection tube requirements for winter ventilation
    """)

    st.subheader("2. Grain Analysis and Processing")
    st.markdown("""
    - **Cereal Grain Analysis**: Determine size parameters, sphericity, and shape characteristics  
    - **Bulk Density & Porosity**: Measure bulk density with different container shapes  
    - **Grain Moisture Content**: Calculate moisture content on wet and dry basis  
    - **Terminal Velocity**: Determine aerodynamic properties for separation
    """)

    st.subheader("3. Post-Harvest Equipment Evaluation")
    st.markdown("""
    - **Screen Cleaner Evaluation**: Analyze cleaning and grading efficiency  
    - **Tray Dryer Evaluation**: Draw drying characteristic curves and evaluate dryer performance  
    - **Belt Conveyor Evaluation**: Calculate capacity and efficiency of horizontal conveyors  
    - **Bucket Conveyor Evaluation**: Analyze vertical conveying systems and their performance
    """)

    st.markdown("""
    Each module includes detailed theory explanations, interactive calculators, and data visualization tools.  
    Select a module from the sidebar to begin exploring these tools.
    """)

    # Contact information
    st.header("Contact Information")

    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 5px solid #1976d2;">
    <p>For suggestions, feedback, or collaboration opportunities, please contact me:</p>
    <ul>
        <li><strong>Phone:</strong> 9626526556</li>
        <li><strong>Email:</strong> vinoth.agritech1115@gmail.com</li>
    </ul>
    <p>I welcome your insights to continually improve and expand these educational tools.</p>
    </div>
    """, unsafe_allow_html=True)





# Summer Cooling System Calculator
elif page == "Summer Cooling System":
    st.markdown("<h2 class='sub-header'>Summer Cooling System Calculator</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Summer cooling in greenhouses is typically done using evaporative cooling systems with cooling pads and exhaust fans.
    This calculator helps determine the required air exchange rate and fan capacity for effective cooling.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Summer Cooling System Theory

        The summer cooling system uses evaporative cooling through wet pads on one side of the greenhouse and exhaust fans on the opposite side.
        As warm air passes through the wet pads, water evaporates and cools the air before it enters the greenhouse.

        The National Greenhouse Manufacturers' Association recommends a standard air exchange rate of **2.5 m³/min/m²** of greenhouse floor area.
        This standard applies under specific conditions:
        - Elevation: 305 m (1,000 ft)
        - Maximum light intensity: 53.8 k lux
        - Temperature rise: 4°C from pad to fans

        For different conditions, correction factors must be applied:

        #### Key Equations:
        """)

        # Formula 1 - using LaTeX for proper display
        st.markdown("**1. Standard Air Removal Rate (Qstd):**")
        st.latex(r"Q_{std} = L \times W \times 2.5")

        st.markdown("""
        Where:
        - L = Greenhouse length (m)
        - W = Greenhouse width (m)
        - 2.5 = Standard air exchange rate (m³/min/m²)
        """)

        # Formula 2 - using LaTeX for proper display
        st.markdown("**2. House Factor (Fhouse):**")
        st.latex(r"F_{house} = F_{elev} \times F_{light} \times F_{temp}")

        st.markdown("""
        Where:
        - F<sub>elev</sub> = Elevation factor
        - F<sub>light</sub> = Light intensity factor
        - F<sub>temp</sub> = Temperature difference factor
        """, unsafe_allow_html=True)

        # Formula 3 - using LaTeX for proper display
        st.markdown("**3. Adjusted Air Removal Rate (Qadj):**")
        st.latex(r"Q_{adj} = Q_{std} \times \max(F_{house}, F_{vel})")

        st.markdown("""
        Where:
        - F<sub>vel</sub> = Velocity factor based on pad-to-fan distance

        The final Q<sub>adj</sub> value determines the required fan capacity. The pad area is calculated based on the air flow rate.
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Elevation Factors (Felev)")
            st.dataframe(elevation_df)

            st.subheader("Light Intensity Factors (Flight)")
            st.dataframe(light_df)

        with col2:
            st.subheader("Temperature Rise Factors (Ftemp)")
            st.dataframe(temp_df)

            st.subheader("Pad-to-Fan Distance Factors (Fvel)")
            combined_vel_df = pd.concat([vel_df1, vel_df2])
            st.dataframe(combined_vel_df)

    # Input parameters
    st.markdown("<h3 class='section-header'>Input Parameters</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Greenhouse Length (m)", min_value=1.0, max_value=200.0, value=30.0, step=0.5)
        width = st.number_input("Greenhouse Width (m)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
        elevation = st.number_input("Elevation Above Sea Level (m)", min_value=0.0, max_value=3000.0, value=915.0,
                                    step=10.0)

    with col2:
        light_intensity = st.number_input("Light Intensity (k lux)", min_value=40.0, max_value=90.0, value=53.8,
                                          step=0.1)
        temp_rise = st.number_input("Temperature Rise from Pad to Fans (°C)", min_value=2.0, max_value=6.0, value=4.0,
                                    step=0.1)
        pad_fan_distance = st.number_input("Pad-to-Fan Distance (m)", min_value=6.0, max_value=40.0, value=30.0,
                                           step=0.5)

    # Calculation
    st.markdown("<h3 class='section-header'>Calculation Results</h3>", unsafe_allow_html=True)

    if st.button("Calculate Summer Cooling Requirements"):
        # Calculate Qstd
        Qstd = length * width * 2.5

        # Get correction factors
        Felev = interpolate_value(elevation_df, 'Elevation (m)', 'Felev', elevation)
        Flight = interpolate_value(light_df, 'Light (k lux)', 'Flight', light_intensity)
        Ftemp = interpolate_value(temp_df, 'Temperature Rise (°C)', 'Ftemp', temp_rise)

        # Calculate Fhouse
        Fhouse = Felev * Flight * Ftemp

        # Get Fvel
        combined_vel_df = pd.concat([vel_df1, vel_df2])
        Fvel = interpolate_value(combined_vel_df, 'Distance (m)', 'Fvel', pad_fan_distance)

        # Calculate Qadj
        Qadj = Qstd * max(Fhouse, Fvel)

        # Calculate pad area assuming 75 m³/min/m² air flow through pad
        pad_area = Qadj / 75

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"**Greenhouse Floor Area:** {length * width:.2f} m²")
            st.markdown(f"**Standard Air Removal Rate (Qstd):** {Qstd:.2f} m³/min")
            st.markdown(f"**Elevation Factor (Felev):** {Felev:.2f}")
            st.markdown(f"**Light Intensity Factor (Flight):** {Flight:.2f}")
            st.markdown(f"**Temperature Factor (Ftemp):** {Ftemp:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"**House Factor (Fhouse):** {Fhouse:.2f}")
            st.markdown(f"**Velocity Factor (Fvel):** {Fvel:.2f}")
            st.markdown(f"**Adjusted Air Removal Rate (Qadj):** {Qadj:.2f} m³/min")
            st.markdown(f"**Required Pad Area:** {pad_area:.2f} m²")
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualization
        st.markdown("<h3 class='section-header'>Visualization</h3>", unsafe_allow_html=True)

        # Create data for bar chart
        factors_df = pd.DataFrame({
            'Factor': ['Elevation', 'Light Intensity', 'Temperature', 'House Factor', 'Velocity Factor'],
            'Value': [Felev, Flight, Ftemp, Fhouse, Fvel]
        })

        # Create bar chart
        chart = alt.Chart(factors_df).mark_bar().encode(
            x=alt.X('Factor', sort=None),
            y='Value',
            color=alt.Color('Factor', legend=None),
            tooltip=['Factor', 'Value']
        ).properties(
            title='Correction Factors',
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # Create a simple diagram of the greenhouse cooling system
        fig, ax = plt.subplots(figsize=(10, 6))
        # Draw a rectangle for the greenhouse
        greenhouse = plt.Rectangle((1, 1), 8, 4, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(greenhouse)

        # Add cooling pads on the left
        pad = plt.Rectangle((0.5, 1.5), 0.5, 3, fill=True, color='blue', alpha=0.5)
        ax.add_patch(pad)

        # Add fans on the right
        fan_positions = [(9.5, 2), (9.5, 4)]
        for x, y in fan_positions:
            circle = plt.Circle((x, y), 0.5, fill=True, color='gray')
            ax.add_patch(circle)

        # Add text and arrows for air flow
        plt.text(2, 0.5, 'Cooling Pads', fontsize=12)
        plt.text(9, 0.5, 'Exhaust Fans', fontsize=12)
        plt.text(5, 3, f'Air Flow: {Qadj:.1f} m³/min', fontsize=12, ha='center')

        # Add arrows for airflow
        for y in range(2, 5):
            ax.arrow(2, y, 6, 0, head_width=0.2, head_length=0.3, fc='black', ec='black')

        # Set limits and remove axes
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Greenhouse Summer Cooling System', fontsize=14)

        st.pyplot(fig)

        st.markdown("""
        <div class='info-box'>
        <b>Fan Selection Note:</b> For this air volume, you would need to select fans with a collective capacity 
        matching the adjusted air removal rate at a static pressure of 30 Pa (0.1 inch). Consult fan performance
        tables to select the appropriate number and size of fans.
        </div>
        """, unsafe_allow_html=True)

# Winter Cooling System Calculator
elif page == "Winter Cooling System":
    st.markdown("<h2 class='sub-header'>Winter Cooling System Calculator</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    Winter cooling in greenhouses typically uses convection tubes to mix outside cooler air with the warmer greenhouse air.
    This calculator helps determine the required air exchange rate and convection tube specifications.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Winter Cooling System Theory

        In winter, the greenhouse interior is typically warmer than the outside air. Cooling is achieved by bringing 
        in cooler outside air through convection tubes with holes that distribute the air evenly throughout the greenhouse.

        Under standard conditions, a volume of **0.61 m³/min** of air should be removed from the greenhouse 
        for each square meter of floor area. The standard conditions are:
        - Elevation: 305 m (1,000 ft)
        - Maximum light intensity: 53.8 k lux
        - Interior-to-exterior temperature difference: Variable

        #### Key Equations:
        """)

        # Formula 1 - using LaTeX for proper display
        st.markdown("**1. Standard Air Volume (Qstd):**")
        st.latex(r"Q_{std} = L \times W \times 0.61")

        st.markdown("""
        Where:
        - L = Greenhouse length (m)
        - W = Greenhouse width (m)
        - 0.61 = Standard winter air exchange rate (m³/min/m²)
        """)

        # Formula 2 - using LaTeX for proper display
        st.markdown("**2. Adjusted Air Volume (Qadj):**")
        st.latex(r"Q_{adj} = Q_{std} \times F_{winter}")

        st.markdown("""
        Where:
        - F<sub>winter</sub> = Winter temperature difference factor

        Based on the adjusted air volume, appropriate convection tubes and their placement can be determined.
        """, unsafe_allow_html=True)

        st.subheader("Winter Temperature Difference Factors (Fwinter)")
        st.dataframe(winter_df)

    # Input parameters
    st.markdown("<h3 class='section-header'>Input Parameters</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Greenhouse Length (m)", min_value=1.0, max_value=200.0, value=30.0, step=0.5)
        width = st.number_input("Greenhouse Width (m)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)

    with col2:
        temp_diff = st.number_input("Temperature Difference (Interior-Exterior) (°C)",
                                    min_value=4.0, max_value=11.0, value=8.0, step=0.1)

    # Calculation
    st.markdown("<h3 class='section-header'>Calculation Results</h3>", unsafe_allow_html=True)

    if st.button("Calculate Winter Cooling Requirements"):
        # Calculate standard air volume
        Qstd = length * width * 0.61

        # Get winter factor
        Fwinter = interpolate_value(winter_df, 'Temperature Difference (°C)', 'Fwinter', temp_diff)

        # Calculate adjusted air volume
        Qadj = Qstd * Fwinter

        # Determine tube requirements based on greenhouse width
        # This is a simplification based on Table 3.7
        if width <= 4.6:
            num_tubes = 1
            tube_diameter = 46 if length <= 30 else 61
        elif width <= 7.6:
            num_tubes = 1
            tube_diameter = 61 if length <= 30 else 76
        elif width <= 10.7:
            num_tubes = 2
            tube_diameter = 61 if length <= 46 else 76
        else:
            num_tubes = 3
            tube_diameter = 76

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"**Greenhouse Floor Area:** {length * width:.2f} m²")
            st.markdown(f"**Standard Air Volume (Qstd):** {Qstd:.2f} m³/min")
            st.markdown(f"**Winter Factor (Fwinter):** {Fwinter:.2f}")
            st.markdown(f"**Adjusted Air Volume (Qadj):** {Qadj:.2f} m³/min")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"**Recommended Number of Tubes:** {num_tubes}")
            st.markdown(f"**Recommended Tube Diameter:** {tube_diameter} cm")
            st.markdown(f"**Air Flow per Tube:** {Qadj / num_tubes:.2f} m³/min")
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualization
        st.markdown("<h3 class='section-header'>Visualization</h3>", unsafe_allow_html=True)

        # Create a chart showing the relationship between temperature difference and winter factor
        chart_data = pd.DataFrame({
            'Temperature Difference (°C)': np.linspace(5.0, 10.0, 20),
            'Winter Factor (Fwinter)': [interpolate_value(winter_df, 'Temperature Difference (°C)', 'Fwinter', temp)
                                        for temp in np.linspace(5.0, 10.0, 20)]
        })

        line_chart = alt.Chart(chart_data).mark_line(color='blue').encode(
            x=alt.X('Temperature Difference (°C)', title='Temperature Difference (°C)'),
            y=alt.Y('Winter Factor (Fwinter)', title='Winter Factor')
        ).properties(
            title='Winter Factor vs. Temperature Difference',
            width=600,
            height=300
        )

        # Add a point for the current temperature difference
        point = alt.Chart(pd.DataFrame({
            'Temperature Difference (°C)': [temp_diff],
            'Winter Factor (Fwinter)': [Fwinter]
        })).mark_circle(size=100, color='red').encode(
            x='Temperature Difference (°C)',
            y='Winter Factor (Fwinter)',
            tooltip=['Temperature Difference (°C)', 'Winter Factor (Fwinter)']
        )

        st.altair_chart(line_chart + point, use_container_width=True)

        # Create a diagram of the greenhouse with convection tubes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw a rectangle for the greenhouse
        greenhouse = plt.Rectangle((1, 1), 8, 4, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(greenhouse)

        # Draw the convection tubes
        tube_y_positions = np.linspace(2, 4, num_tubes)
        for y in tube_y_positions:
            tube = plt.Rectangle((1.5, y - 0.1), 7, 0.2, fill=True, color='lightblue', alpha=0.8)
            ax.add_patch(tube)

            # Add holes to the tube
            for x in np.linspace(2, 8, 12):
                hole = plt.Circle((x, y), 0.1, fill=True, color='white')
                ax.add_patch(hole)

        # Add fan on the left
        fan = plt.Rectangle((0.5, 2.5), 0.5, 1, fill=True, color='gray')
        ax.add_patch(fan)

        # Add text
        plt.text(0.7, 3.7, 'Fan', fontsize=10, ha='center')
        for i, y in enumerate(tube_y_positions):
            plt.text(5, y - 0.3, f'Tube {i + 1}: {tube_diameter} cm', fontsize=10, ha='center')

        plt.text(5, 0.5, f'Total Air Flow: {Qadj:.1f} m³/min', fontsize=12, ha='center')

        # Set limits and remove axes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Greenhouse Winter Cooling System', fontsize=14)

        st.pyplot(fig)

        st.markdown("""
        <div class='info-box'>
        <b>Installation Note:</b> Convection tubes should be installed at plant height if possible, and 
        should run the length of the greenhouse. The holes in the tubes should be spaced to provide even 
        air distribution, with pairs of holes on opposite sides of the tube.
        </div>
        """, unsafe_allow_html=True)

# Cereal Grain Analysis
elif page == "Cereal Grain Analysis":
    st.markdown("<h2 class='sub-header'>Cereal Grain Size and Shape Analysis</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This tool helps analyze the physical characteristics of cereal grains, including size measurements,
    sphericity, roundness, and shape identification - important parameters for grain processing and handling.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Size and Shape Analysis Theory

        Size and shape are critical physical properties of cereal grains that affect handling, processing,
        and quality assessment. The following parameters are measured:

        #### Size Parameters:
        - **Length**: The longest dimension of the grain
        - **Breadth/Width**: The dimension perpendicular to length
        - **Thickness/Height**: The dimension perpendicular to both length and breadth
        - **Diameter**: Used for circular or spherical objects

        #### Shape Parameters:
        """)

        # Formula 1: Sphericity basic definition
        st.markdown("**1. Sphericity:**")
        st.markdown(
            "Sphericity is defined as the ratio of the surface area of a sphere having the same volume as the particle to the surface area of the particle, or alternatively:")

        st.latex(r"\text{Sphericity} = \frac{D_e}{D_c}")

        st.markdown("""
        Where:
        - D<sub>e</sub> = Diameter of a sphere of the same volume
        - D<sub>c</sub> = Diameter of the smallest circumscribing sphere (typically the largest dimension)
        """, unsafe_allow_html=True)

        st.markdown("It can also be calculated using the three principal dimensions:")

        st.latex(
            r"\text{Sphericity} = \left[\frac{V_p}{V_c}\right]^{1/3} = \frac{[\frac{\pi}{6} (l \cdot b \cdot t)^{1/3}]}{\frac{\pi}{6} l^3}")

        st.markdown("""
        Where:
        - l = length
        - b = breadth
        - t = thickness
        """)

        # Formula 2: Roundness
        st.markdown("**2. Roundness:**")
        st.markdown("Roundness measures the sharpness of the corners of a particle, defined as:")

        st.latex(
            r"\text{Roundness} = \frac{\text{Largest projected area in natural rest position } (A_p)}{\text{Area of smallest circumscribing circle } (A_c)}")

        # Formula 3: Roundness ratio
        st.markdown("**3. Roundness Ratio:**")

        st.latex(
            r"\text{Roundness ratio} = \frac{\text{Radius of curvature of sharpest corner } (r)}{\text{Mean radius of the particle } (R)}")

        # Show the shape table
        st.markdown("### Common Grain/Fruit Shapes:")
        shapes = [
            {"Shape": "Round", "Description": "Approaching spheroid like.", "Examples": "Lemon, apple, etc."},
            {"Shape": "Oblate", "Description": "Flattened at stem end apex", "Examples": "Pomegranate, pumpkin"},
            {"Shape": "Oblong", "Description": "Vertical diameter greater than the horizontal diameter.",
             "Examples": "Ashguard."},
            {"Shape": "Conic", "Description": "Tapered towards the apex.", "Examples": "Radish, carrot."},
            {"Shape": "Ovate", "Description": "Egg shaped and broad at the stem end.",
             "Examples": "Mango, brinjal etc."},
            {"Shape": "Obovate", "Description": "Inverted ovate.", "Examples": "Cashew fruit"},
            {"Shape": "Elliptical", "Description": "Approaching ellipsoid.", "Examples": "Guava etc."},
            {"Shape": "Truncate", "Description": "Having both ends flattened and squared.",
             "Examples": "Apple, orange etc."},
            {"Shape": "Unequal", "Description": "One half portion larger than the another half portion.",
             "Examples": "Mango"},
            {"Shape": "Ribbed", "Description": "In cross-section, sides are more or less angular.",
             "Examples": "Bitter gourd, ribbed gourd."},
            {"Shape": "Regular", "Description": "Horizontal section approaches to a circle.", "Examples": "Grapes."},
            {"Shape": "Irregular", "Description": "Horizontal cross-section departs from a circle.",
             "Examples": "Mango, ladies finger, capsicum, etc."}
        ]
        shape_df = pd.DataFrame(shapes)
        st.dataframe(shape_df)

        st.markdown("""
        ### Importance of Grain Shape and Size Analysis:

        1. **Packaging Systems**: Designing appropriate packaging for different grain types
        2. **Sieve Selection**: Determining proper sieve perforations for grading and cleaning
        3. **Processing Equipment**: Designing handling and processing equipment
        4. **Storage**: Calculating storage requirements and bulk density
        5. **Quality Assessment**: Evaluating grain quality based on physical characteristics

        ### Measurement Procedure:

        1. Take minimum 10 grains/seeds from each lot of particular moisture content
        2. Measure length, breadth, and thickness using vernier caliper, screw gauge, or traveling microscope
        3. Calculate the mean of at least 25 grains/seeds
        4. For roundness and roundness ratio, trace the projection of the grain and measure the relevant areas and radii
        """)

        # Show diagrams
        st.image("https://via.placeholder.com/800x300.png?text=Grain+Measurement+Diagrams",
                 caption="Measurement diagrams for grain analysis")

    # Measurements input
    st.markdown("<h3 class='section-header'>Grain Measurements</h3>", unsafe_allow_html=True)

    grain_type = st.text_input("Grain/Seed Type (Variety)", "")

    # Create a tabbed interface for different input methods
    tab1, tab2, tab3 = st.tabs(["Single Grain Analysis", "Multiple Samples", "Shape Identification"])

    with tab1:
        st.markdown("Enter measurements for an individual grain sample:")

        # Create a form for entering measurements
        with st.form("grain_measurements_form"):
            col1, col2 = st.columns(2)

            with col1:
                length = st.number_input("Length (mm)", min_value=0.0, step=0.01, format="%.2f")
                breadth = st.number_input("Breadth (mm)", min_value=0.0, step=0.01, format="%.2f")
                thickness = st.number_input("Thickness (mm)", min_value=0.0, step=0.01, format="%.2f")

            with col2:
                # Optional for roundness calculations
                proj_area = st.number_input("Projected Area (mm²) (Optional)", min_value=0.0, step=0.01, format="%.2f")
                circ_area = st.number_input("Circumscribing Circle Area (mm²) (Optional)", min_value=0.0, step=0.01,
                                            format="%.2f")
                corner_radius = st.number_input("Radius of Sharpest Corner (mm) (Optional)", min_value=0.0, step=0.01,
                                                format="%.2f")
                mean_radius = st.number_input("Mean Radius of Particle (mm) (Optional)", min_value=0.0, step=0.01,
                                              format="%.2f")

            calculate_button = st.form_submit_button("Calculate Parameters")

        if calculate_button and length > 0 and breadth > 0 and thickness > 0:
            # Calculate sphericity
            volume = (np.pi / 6) * length * breadth * thickness
            equiv_diameter = (volume * 6 / np.pi) ** (1 / 3)
            sphericity = ((np.pi / 6) * (length * breadth * thickness)) ** (1 / 3) / ((np.pi / 6) * length ** 3) ** (
                        1 / 3)
            # Simplified formula
            simplified_sphericity = ((length * breadth * thickness) ** (1 / 3)) / length

            # Calculate roundness if areas are provided
            roundness = None
            if proj_area > 0 and circ_area > 0:
                roundness = proj_area / circ_area

            # Calculate roundness ratio if radii are provided
            roundness_ratio = None
            if corner_radius > 0 and mean_radius > 0:
                roundness_ratio = corner_radius / mean_radius

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Grain Type:** {grain_type}")
                st.markdown(f"**Length:** {length:.2f} mm")
                st.markdown(f"**Breadth:** {breadth:.2f} mm")
                st.markdown(f"**Thickness:** {thickness:.2f} mm")
                st.markdown(f"**Volume:** {volume:.2f} mm³")

            with col2:
                st.markdown(f"**Equivalent Diameter:** {equiv_diameter:.2f} mm")
                st.markdown(f"**Sphericity:** {sphericity:.4f}")
                if roundness:
                    st.markdown(f"**Roundness:** {roundness:.4f}")
                if roundness_ratio:
                    st.markdown(f"**Roundness Ratio:** {roundness_ratio:.4f}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Suggest shape based on dimensions
            l_b_ratio = length / breadth
            b_t_ratio = breadth / thickness

            st.markdown("<h4>Possible Shape Classification:</h4>", unsafe_allow_html=True)

            if 0.9 <= l_b_ratio <= 1.1 and 0.9 <= b_t_ratio <= 1.1:
                shape = "Round (approaching spheroid)"
            elif l_b_ratio > 1.5 and 0.9 <= b_t_ratio <= 1.1:
                shape = "Oblong (length significantly greater than width)"
            elif l_b_ratio < 0.85:
                shape = "Oblate (flattened)"
            elif l_b_ratio > 1.1 and b_t_ratio > 1.1:
                shape = "Elliptical (approaching ellipsoid)"
            else:
                shape = "Irregular"

            st.markdown(f"Based on the dimensions, this grain appears to be: **{shape}**")

            # Visualization
            st.markdown("<h4>Visualization:</h4>", unsafe_allow_html=True)

            # Create a simple 3D visualization of the grain
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(121, projection='3d')

            # Create an ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = (length / 2) * np.outer(np.cos(u), np.sin(v))
            y = (breadth / 2) * np.outer(np.sin(u), np.sin(v))
            z = (thickness / 2) * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color='wheat', alpha=0.8)

            # Set limits and labels
            ax.set_xlim(-length / 2, length / 2)
            ax.set_ylim(-breadth / 2, breadth / 2)
            ax.set_zlim(-thickness / 2, thickness / 2)
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Breadth (mm)')
            ax.set_zlabel('Thickness (mm)')
            ax.set_title('3D Model of Grain')

            # Add a 2D projection diagram
            ax2 = fig.add_subplot(122)
            theta = np.linspace(0, 2 * np.pi, 100)

            # Draw ellipse for top view
            x_ellipse = (length / 2) * np.cos(theta)
            y_ellipse = (breadth / 2) * np.sin(theta)
            ax2.plot(x_ellipse, y_ellipse, color='brown')
            ax2.fill(x_ellipse, y_ellipse, color='wheat', alpha=0.6)

            # Draw circumscribing circle
            radius = max(length / 2, breadth / 2)
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            ax2.plot(x_circle, y_circle, 'k--', alpha=0.5)

            ax2.set_aspect('equal')
            ax2.set_xlabel('Length (mm)')
            ax2.set_ylabel('Breadth (mm)')
            ax2.set_title('Top View with Circumscribing Circle')
            ax2.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        st.markdown("""
        ### Multiple Grain Sample Analysis

        Enter measurements for multiple grain samples to calculate average parameters and statistics.
        """)

        # Allow user to set number of samples
        num_samples = st.number_input("Number of samples", min_value=1, max_value=25, value=5, step=1)

        # Create input fields for multiple samples
        measurements = []
        st.markdown("<h4>Enter measurements for each sample:</h4>", unsafe_allow_html=True)

        use_sample_data = st.checkbox("Use sample data for demonstration")

        if use_sample_data:
            # Generate sample data for demonstration
            np.random.seed(42)  # For reproducibility
            sample_lengths = np.random.normal(8.5, 0.5, num_samples)
            sample_breadths = np.random.normal(4.2, 0.3, num_samples)
            sample_thicknesses = np.random.normal(3.0, 0.2, num_samples)

            # Format sample data for display
            sample_data = []
            for i in range(num_samples):
                sample_data.append({
                    'Sample': i + 1,
                    'Length': round(sample_lengths[i], 2),
                    'Breadth': round(sample_breadths[i], 2),
                    'Thickness': round(sample_thicknesses[i], 2)
                })

            # Display sample data
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df)

            measurements = sample_data

        else:
            # Create a form for manual data entry
            with st.form("multiple_samples_form"):
                for i in range(num_samples):
                    st.markdown(f"**Sample {i + 1}**")
                    cols = st.columns(3)
                    length = cols[0].number_input(f"Length {i + 1} (mm)", min_value=0.0, step=0.01, format="%.2f",
                                                  key=f"length_{i}")
                    breadth = cols[1].number_input(f"Breadth {i + 1} (mm)", min_value=0.0, step=0.01, format="%.2f",
                                                   key=f"breadth_{i}")
                    thickness = cols[2].number_input(f"Thickness {i + 1} (mm)", min_value=0.0, step=0.01, format="%.2f",
                                                     key=f"thickness_{i}")

                    measurements.append({
                        'Sample': i + 1,
                        'Length': length,
                        'Breadth': breadth,
                        'Thickness': thickness
                    })

                calculate_button = st.form_submit_button("Calculate Statistics")

                if calculate_button:
                    pass  # The calculation will be done below

        # Process the measurements if they exist
        if measurements and all(m['Length'] > 0 and m['Breadth'] > 0 and m['Thickness'] > 0 for m in measurements):
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(measurements)

            # Calculate sphericity for each sample
            sphericities = []
            for _, row in df.iterrows():
                length = row['Length']
                breadth = row['Breadth']
                thickness = row['Thickness']

                # Simplified sphericity formula
                sphericity = ((length * breadth * thickness) ** (1 / 3)) / length
                sphericities.append(sphericity)

            df['Sphericity'] = sphericities

            # Calculate statistics
            stats_df = pd.DataFrame({
                'Parameter': ['Length (mm)', 'Breadth (mm)', 'Thickness (mm)', 'Sphericity'],
                'Mean': [df['Length'].mean(), df['Breadth'].mean(), df['Thickness'].mean(), df['Sphericity'].mean()],
                'Min': [df['Length'].min(), df['Breadth'].min(), df['Thickness'].min(), df['Sphericity'].min()],
                'Max': [df['Length'].max(), df['Breadth'].max(), df['Thickness'].max(), df['Sphericity'].max()],
                'Std Dev': [df['Length'].std(), df['Breadth'].std(), df['Thickness'].std(), df['Sphericity'].std()]
            })

            # Display statistics
            st.markdown("<h4>Statistical Summary:</h4>", unsafe_allow_html=True)
            st.dataframe(stats_df.style.format({
                'Mean': '{:.2f}',
                'Min': '{:.2f}',
                'Max': '{:.2f}',
                'Std Dev': '{:.2f}'
            }))

            # Display all measurements with calculated sphericity
            st.markdown("<h4>All Measurements with Calculated Parameters:</h4>", unsafe_allow_html=True)
            st.dataframe(df.style.format({
                'Length': '{:.2f}',
                'Breadth': '{:.2f}',
                'Thickness': '{:.2f}',
                'Sphericity': '{:.4f}'
            }))

            # Visualizations
            st.markdown("<h4>Visualizations:</h4>", unsafe_allow_html=True)

            # Box plots for dimensions
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create box plot data
            box_data = [df['Length'], df['Breadth'], df['Thickness']]
            box_labels = ['Length', 'Breadth', 'Thickness']

            ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       whiskerprops=dict(color='blue'),
                       capprops=dict(color='blue'),
                       medianprops=dict(color='darkblue'))

            ax.set_ylabel('Dimension (mm)')
            ax.set_title('Distribution of Grain Dimensions')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            st.pyplot(fig)

            # Sphericity histogram
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            ax2.hist(df['Sphericity'], bins=10, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(df['Sphericity'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df["Sphericity"].mean():.4f}')

            ax2.set_xlabel('Sphericity')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Sphericity Values')
            ax2.legend()
            ax2.grid(linestyle='--', alpha=0.7)

            st.pyplot(fig2)

            # 3D scatter plot of dimensions
            fig3 = plt.figure(figsize=(10, 8))
            ax3 = fig3.add_subplot(111, projection='3d')

            ax3.scatter(df['Length'], df['Breadth'], df['Thickness'], c='gold', s=100, marker='o', edgecolor='black')

            ax3.set_xlabel('Length (mm)')
            ax3.set_ylabel('Breadth (mm)')
            ax3.set_zlabel('Thickness (mm)')
            ax3.set_title('3D Scatter Plot of Grain Dimensions')

            # Highlight the mean point
            mean_length = df['Length'].mean()
            mean_breadth = df['Breadth'].mean()
            mean_thickness = df['Thickness'].mean()

            ax3.scatter([mean_length], [mean_breadth], [mean_thickness], c='red', s=200, marker='*', edgecolor='black')
            ax3.text(mean_length, mean_breadth, mean_thickness, 'Mean', color='red')

            st.pyplot(fig3)

            # Calculate average shape classification
            avg_length = df['Length'].mean()
            avg_breadth = df['Breadth'].mean()
            avg_thickness = df['Thickness'].mean()

            l_b_ratio = avg_length / avg_breadth
            b_t_ratio = avg_breadth / avg_thickness

            if 0.9 <= l_b_ratio <= 1.1 and 0.9 <= b_t_ratio <= 1.1:
                avg_shape = "Round (approaching spheroid)"
            elif l_b_ratio > 1.5 and 0.9 <= b_t_ratio <= 1.1:
                avg_shape = "Oblong (length significantly greater than width)"
            elif l_b_ratio < 0.85:
                avg_shape = "Oblate (flattened)"
            elif l_b_ratio > 1.1 and b_t_ratio > 1.1:
                avg_shape = "Elliptical (approaching ellipsoid)"
            else:
                avg_shape = "Irregular"

            st.markdown(
                f"<div class='result-box'>Based on the average dimensions, this grain population appears to be: <b>{avg_shape}</b></div>",
                unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        ### Grain Shape Identification

        Use this tool to identify the shape of grain or fruit samples based on visual characteristics.
        """)

        # Create a visual selection tool
        st.markdown("<h4>Select the shape that most closely resembles your sample:</h4>", unsafe_allow_html=True)

        # Create a grid of shape options with images and descriptions
        col1, col2, col3 = st.columns(3)

        shape_selected = None

        with col1:
            if st.button("Round", key="shape_round"):
                shape_selected = "Round"
            st.image("https://via.placeholder.com/150x150.png?text=Round", width=150)
            st.markdown("Approaching spheroid like.")

            if st.button("Conic", key="shape_conic"):
                shape_selected = "Conic"
            st.image("https://via.placeholder.com/150x150.png?text=Conic", width=150)
            st.markdown("Tapered towards the apex.")

            if st.button("Truncate", key="shape_truncate"):
                shape_selected = "Truncate"
            st.image("https://via.placeholder.com/150x150.png?text=Truncate", width=150)
            st.markdown("Having both ends flattened and squared.")

        with col2:
            if st.button("Oblate", key="shape_oblate"):
                shape_selected = "Oblate"
            st.image("https://via.placeholder.com/150x150.png?text=Oblate", width=150)
            st.markdown("Flattened at stem end apex.")

            if st.button("Ovate", key="shape_ovate"):
                shape_selected = "Ovate"
            st.image("https://via.placeholder.com/150x150.png?text=Ovate", width=150)
            st.markdown("Egg shaped and broad at the stem end.")

            if st.button("Unequal", key="shape_unequal"):
                shape_selected = "Unequal"
            st.image("https://via.placeholder.com/150x150.png?text=Unequal", width=150)
            st.markdown("One half portion larger than another half.")

        with col3:
            if st.button("Oblong", key="shape_oblong"):
                shape_selected = "Oblong"
            st.image("https://via.placeholder.com/150x150.png?text=Oblong", width=150)
            st.markdown("Vertical diameter greater than horizontal.")

            if st.button("Elliptical", key="shape_elliptical"):
                shape_selected = "Elliptical"
            st.image("https://via.placeholder.com/150x150.png?text=Elliptical", width=150)
            st.markdown("Approaching ellipsoid.")

            if st.button("Ribbed", key="shape_ribbed"):
                shape_selected = "Ribbed"
            st.image("https://via.placeholder.com/150x150.png?text=Ribbed", width=150)
            st.markdown("In cross-section, sides are more or less angular.")

        # Allow manual selection as well
        manual_shape = st.selectbox(
            "Or select shape from dropdown:",
            ["", "Round", "Oblate", "Oblong", "Conic", "Ovate", "Obovate", "Elliptical", "Truncate", "Unequal",
             "Ribbed", "Regular", "Irregular"]
        )

        if manual_shape:
            shape_selected = manual_shape

        # Display information about the selected shape
        if shape_selected:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Selected Shape: {shape_selected}</h3>", unsafe_allow_html=True)

            # Find the shape info from our list of dictionaries
            shape_info = next((s for s in shapes if s["Shape"] == shape_selected), None)

            if shape_info:
                st.markdown(f"**Description:** {shape_info['Description']}")
                st.markdown(f"**Examples:** {shape_info['Examples']}")
            else:
                st.markdown("**Description:** Information not available")
                st.markdown("**Examples:** Information not available")

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""
            ### Measurement Recommendations

            Based on the selected shape, here are the key measurements you should focus on:
            """)

            if shape_selected in ["Round"]:
                st.markdown("""
                - Measure diameter in multiple directions to check for uniformity
                - Focus on sphericity calculation (values close to 1.0 indicate spherical shape)
                """)
            elif shape_selected in ["Elliptical", "Oblong"]:
                st.markdown("""
                - Carefully measure the major axis (length) and minor axis (breadth)
                - Calculate length/breadth ratio to quantify elongation
                - For oblong shapes, ensure vertical vs. horizontal orientation is consistent
                """)
            elif shape_selected in ["Conic", "Ovate", "Obovate"]:
                st.markdown("""
                - Measure both at the widest point and at regular intervals along the length
                - Identify the broader end and narrower end
                - Calculate tapering ratio between widest and narrowest parts
                """)
            elif shape_selected in ["Ribbed", "Irregular"]:
                st.markdown("""
                - Take multiple cross-sectional measurements
                - Measure the minimum and maximum dimensions at each cross-section
                - Calculate standard deviation of measurements to quantify irregularity
                """)

            st.markdown("""
            ### Processing Implications

            The identified shape has these implications for processing and handling:
            """)

            if shape_selected in ["Round", "Oblate"]:
                st.markdown("""
                - Rolls easily on processing lines
                - Usually requires round perforations in grading sieves
                - Typically good bulk handling properties
                """)
            elif shape_selected in ["Elliptical", "Oblong", "Conic"]:
                st.markdown("""
                - May require orientation during processing
                - Often uses elongated perforations in grading sieves
                - Can be more susceptible to mechanical damage on processing lines
                """)
            elif shape_selected in ["Ribbed", "Irregular", "Unequal"]:
                st.markdown("""
                - More challenging to process mechanically
                - May require specialized handling equipment
                - Typically more difficult to grade uniformly
                """)

# Bulk Density & Porosity Module
elif page == "Bulk Density & Porosity":
    st.markdown("<h2 class='sub-header'>Bulk Density and Porosity of Biomaterials</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This tool helps determine key physical properties of cereal grains and seeds including bulk density,
    porosity, and true density - critical parameters for processing, handling, and storage of agricultural materials.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Bulk Density, Porosity, and True Density Theory

        These physical properties are fundamental for the design of handling, processing, and storage systems for grains and seeds.

        #### Key Concepts:

        **1. Bulk Density:**
        - Definition: Mass of grain/seed per unit volume, including void spaces
        - Units: kg/m³ or g/cc
        - Importance: Affects storage capacity, container design, and transport calculations

        **2. True Density (Particle Density):**
        - Definition: Mass of grain/seed per unit volume, excluding void spaces (pores)
        - Units: kg/m³ or g/cc
        - Importance: Represents the actual density of the grain material itself

        **3. Porosity:**
        - Definition: Percentage of void volume in the grain/seed mass
        - Units: Percentage (%)
        - Importance: Affects aeration, drying, and fluid flow through grain mass
        """)

        st.markdown("#### Measurement Methods:")

        st.markdown("""
        **Bulk Density Measurement:**
        - Fill a container of known volume with grain/seed
        - Weigh the filled container
        - Calculate:
        """)

        st.latex(r"\text{Bulk Density} = \frac{\text{Mass of grain}}{\text{Volume of container}}")

        st.markdown("""
        **Porosity Measurement:**
        - Using air pressure differential to determine void space
        - Based on the principle that air will fill the void spaces in the grain
        - Using the formula:
        """)

        st.latex(r"\text{Porosity} = \left[\frac{P_1-P_2}{P_2}\right] \times 100")

        st.markdown("""
        Where P<sub>1</sub> is initial pressure and P<sub>2</sub> is equilibrium pressure
        """, unsafe_allow_html=True)

        st.markdown("""
        **True Density Calculation:**
        - Calculated using bulk density and porosity
        - Formula:
        """)

        st.latex(r"\text{True Density} = \frac{\text{Bulk Density}}{(1 - \text{Porosity})}")

        st.markdown("Note: Porosity must be expressed as a decimal, not percentage")

        st.markdown("#### The Physics Behind Porosity Measurement:")

        st.markdown("The measurement uses the gas law equation:")

        st.latex(r"P_1 V_1 = MRT")

        st.markdown("""
        Where:
        - P = Pressure
        - V = Volume
        - M = Mass of air
        - R = Gas constant
        - T = Temperature
        """)

        st.markdown("When air from the air tank is allowed to occupy the pore space in the grain tank:")

        st.latex(r"M = M_1 + M_2")

        st.latex(r"M = \frac{P_2 V_1}{RT} + \frac{P_2 V_2}{RT} = \frac{P_1 V_1}{RT}")

        st.markdown("This leads to the equation for calculating porosity:")

        st.latex(r"\frac{V_2}{V_1} = \frac{P_1-P_2}{P_2} \times 100")
        # Show the porosity apparatus diagram
        st.image("https://via.placeholder.com/800x400.png?text=Porosity+Apparatus+Diagram",
                 caption="Fig 5.1: Apparatus for determination of porosity of grains")

    # Create tabs for different operations
    tab1, tab2, tab3 = st.tabs(["Bulk Density Calculator", "Porosity Calculator", "True Density Calculator"])

    with tab1:
        st.markdown("<h3 class='section-header'>Bulk Density Measurement</h3>", unsafe_allow_html=True)

        st.markdown("""
        Bulk density is measured by filling a container of known volume with grain/seed and measuring the mass.
        Enter your measurements below:
        """)

        # Container specifications
        with st.form("bulk_density_form"):
            grain_type = st.text_input("Grain/Seed Type", "")
            moisture_content = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0, value=14.0,
                                               step=0.1)

            st.markdown("#### Container Dimensions:")

            container_shape = st.selectbox("Container Shape", ["Cylindrical", "Rectangular", "Custom Volume"])

            if container_shape == "Cylindrical":
                col1, col2 = st.columns(2)
                with col1:
                    container_diameter = st.number_input("Container Diameter (cm)", min_value=0.1, value=10.0, step=0.1)
                with col2:
                    container_height = st.number_input("Container Height (cm)", min_value=0.1, value=15.0, step=0.1)
                container_volume = np.pi * (container_diameter / 2) ** 2 * container_height

            elif container_shape == "Rectangular":
                col1, col2, col3 = st.columns(3)
                with col1:
                    container_length = st.number_input("Container Length (cm)", min_value=0.1, value=10.0, step=0.1)
                with col2:
                    container_width = st.number_input("Container Width (cm)", min_value=0.1, value=10.0, step=0.1)
                with col3:
                    container_height_rect = st.number_input("Container Height (cm)", min_value=0.1, value=10.0,
                                                            step=0.1)
                container_volume = container_length * container_width * container_height_rect

            else:  # Custom Volume
                container_volume = st.number_input("Container Volume (cc or cm³)", min_value=0.1, value=1000.0,
                                                   step=0.1)

            st.markdown("#### Mass Measurements:")

            empty_container_mass = st.number_input("Mass of Empty Container (g)", min_value=0.0, value=50.0, step=0.1)
            filled_container_mass = st.number_input("Mass of Container + Sample (g)", min_value=0.0, value=800.0,
                                                    step=0.1)

            # Number of replications
            num_replications = st.number_input("Number of Replications", min_value=1, max_value=10, value=3, step=1)

            # Add multiple replication measurements if needed
            replication_data = []
            if num_replications > 1:
                st.markdown("#### Additional Replications:")
                for i in range(1, num_replications):
                    col1, col2 = st.columns(2)
                    with col1:
                        rep_empty_mass = st.number_input(f"Empty Container Mass - Rep {i + 1} (g)",
                                                         min_value=0.0, value=50.0, step=0.1, key=f"empty_mass_{i}")
                    with col2:
                        rep_filled_mass = st.number_input(f"Filled Container Mass - Rep {i + 1} (g)",
                                                          min_value=0.0, value=800.0, step=0.1, key=f"filled_mass_{i}")

                    replication_data.append((rep_empty_mass, rep_filled_mass))

            submit_button = st.form_submit_button("Calculate Bulk Density")

        if submit_button:
            # Calculate sample mass and bulk density for first measurement
            sample_mass = filled_container_mass - empty_container_mass
            bulk_density = sample_mass / container_volume  # g/cc

            # Calculate for replications if any
            all_bulk_densities = [bulk_density]
            all_sample_masses = [sample_mass]

            for empty_mass, filled_mass in replication_data:
                rep_sample_mass = filled_mass - empty_mass
                rep_bulk_density = rep_sample_mass / container_volume
                all_bulk_densities.append(rep_bulk_density)
                all_sample_masses.append(rep_sample_mass)

            # Calculate average
            avg_bulk_density = sum(all_bulk_densities) / len(all_bulk_densities)
            avg_sample_mass = sum(all_sample_masses) / len(all_sample_masses)

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(f"### Bulk Density Results for {grain_type}")
            st.markdown(f"**Moisture Content:** {moisture_content:.1f}%")
            st.markdown(f"**Container Volume:** {container_volume:.2f} cc (cm³)")

            # Create a table of results
            results_data = []
            for i in range(len(all_bulk_densities)):
                rep_name = "Primary Measurement" if i == 0 else f"Replication {i}"
                results_data.append({
                    "Replication": rep_name,
                    "Sample Mass (g)": round(all_sample_masses[i], 2),
                    "Bulk Density (g/cc)": round(all_bulk_densities[i], 4),
                    "Bulk Density (kg/m³)": round(all_bulk_densities[i] * 1000, 2)
                })

            results_df = pd.DataFrame(results_data)
            st.table(results_df)

            st.markdown(f"**Average Bulk Density:** {avg_bulk_density:.4f} g/cc ({avg_bulk_density * 1000:.2f} kg/m³)")
            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization
            if len(all_bulk_densities) > 1:
                st.markdown("<h4>Visualization of Measurements:</h4>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 6))

                reps = ["Primary"] + [f"Rep {i + 1}" for i in range(len(replication_data))]
                ax.bar(reps, all_bulk_densities, color='skyblue', edgecolor='navy')

                # Add a horizontal line for the average
                ax.axhline(y=avg_bulk_density, color='red', linestyle='--',
                           label=f'Average: {avg_bulk_density:.4f} g/cc')

                ax.set_ylabel('Bulk Density (g/cc)')
                ax.set_title(f'Bulk Density Measurements for {grain_type}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

            # Explanation of results
            st.markdown("""
            ### Interpretation of Bulk Density Results

            Bulk density is an important characteristic for:

            1. **Storage calculations**: Higher bulk density means more grain can be stored in the same volume
            2. **Transport considerations**: Affects weight distribution in transport vehicles
            3. **Processing equipment design**: Affects flow characteristics through equipment
            4. **Quality indicator**: Can indicate grain maturity, soundness, and cleanliness

            Typical bulk density ranges:
            - Wheat: 720-830 kg/m³
            - Rice (rough): 560-610 kg/m³
            - Corn: 720-800 kg/m³
            - Soybeans: 700-750 kg/m³
            """)

    with tab2:
        st.markdown("<h3 class='section-header'>Porosity Measurement</h3>", unsafe_allow_html=True)

        st.markdown("""
        Porosity is the percentage of void space in a grain/seed mass. It can be measured using an air comparison pycnometer
        or calculated from pressure measurements as shown in the porosity apparatus.
        """)

        # Description of the apparatus
        st.markdown("""
        The porosity apparatus has two identical glass jars:
        - Air tank (F): For holding initial pressure of air
        - Grain tank (A): Filled with the grain sample

        The procedure involves measuring:
        - Initial pressure (P₁) in the air tank
        - Equilibrium pressure (P₂) after allowing air to fill the void spaces in the grain tank
        """)

        with st.form("porosity_form"):
            grain_type_porosity = st.text_input("Grain/Seed Type", "")
            moisture_content_porosity = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0,
                                                        value=14.0, step=0.1)

            # Pressure measurements
            col1, col2 = st.columns(2)
            with col1:
                initial_pressure = st.number_input("Initial Pressure (P₁) (cm of water)", min_value=0.1, value=20.0,
                                                   step=0.1)
            with col2:
                final_pressure = st.number_input("Equilibrium Pressure (P₂) (cm of water)", min_value=0.1, value=16.0,
                                                 step=0.1)

            # Additional replications
            num_replications_porosity = st.number_input("Number of Replications", min_value=1, max_value=10, value=3,
                                                        step=1)

            replication_data_porosity = []
            if num_replications_porosity > 1:
                st.markdown("#### Additional Replications:")
                for i in range(1, num_replications_porosity):
                    col1, col2 = st.columns(2)
                    with col1:
                        rep_initial_pressure = st.number_input(f"Initial Pressure (P₁) - Rep {i + 1}",
                                                               min_value=0.1, value=20.0, step=0.1, key=f"p1_{i}")
                    with col2:
                        rep_final_pressure = st.number_input(f"Equilibrium Pressure (P₂) - Rep {i + 1}",
                                                             min_value=0.1, value=16.0, step=0.1, key=f"p2_{i}")

                    replication_data_porosity.append((rep_initial_pressure, rep_final_pressure))

            submit_button_porosity = st.form_submit_button("Calculate Porosity")

        if submit_button_porosity:
            # Calculate porosity for first measurement
            porosity = ((initial_pressure - final_pressure) / final_pressure) * 100  # percentage

            # Calculate for replications if any
            all_porosities = [porosity]

            for p1, p2 in replication_data_porosity:
                rep_porosity = ((p1 - p2) / p2) * 100
                all_porosities.append(rep_porosity)

            # Calculate average
            avg_porosity = sum(all_porosities) / len(all_porosities)

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(f"### Porosity Results for {grain_type_porosity}")
            st.markdown(f"**Moisture Content:** {moisture_content_porosity:.1f}%")

            # Create a table of results
            results_data_porosity = []
            for i in range(len(all_porosities)):
                rep_name = "Primary Measurement" if i == 0 else f"Replication {i}"

                # Get the pressure values
                if i == 0:
                    p1, p2 = initial_pressure, final_pressure
                else:
                    p1, p2 = replication_data_porosity[i - 1]

                results_data_porosity.append({
                    "Replication": rep_name,
                    "Initial Pressure P₁ (cm)": p1,
                    "Equilibrium Pressure P₂ (cm)": p2,
                    "Porosity (%)": round(all_porosities[i], 2)
                })

            results_df_porosity = pd.DataFrame(results_data_porosity)
            st.table(results_df_porosity)

            st.markdown(f"**Average Porosity:** {avg_porosity:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization
            if len(all_porosities) > 1:
                st.markdown("<h4>Visualization of Measurements:</h4>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 6))

                reps = ["Primary"] + [f"Rep {i + 1}" for i in range(len(replication_data_porosity))]
                ax.bar(reps, all_porosities, color='lightgreen', edgecolor='darkgreen')

                # Add a horizontal line for the average
                ax.axhline(y=avg_porosity, color='red', linestyle='--', label=f'Average: {avg_porosity:.2f}%')

                ax.set_ylabel('Porosity (%)')
                ax.set_title(f'Porosity Measurements for {grain_type_porosity}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

            # Explanation of results
            st.markdown("""
            ### Interpretation of Porosity Results

            Porosity is critical for:

            1. **Aeration**: Higher porosity allows better air circulation through grain mass
            2. **Drying**: Affects drying rate and efficiency
            3. **Fluid flow**: Determines how air, water, or fumigants move through grain
            4. **Bulk storage**: Affects pressure distribution in silos and storage bins

            Typical porosity ranges:
            - Wheat: 38-45%
            - Rice: 45-55%
            - Corn: 35-45%
            - Soybeans: 40-45%
            """)

    with tab3:
        st.markdown("<h3 class='section-header'>True Density Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        True density (particle density) is the density of the grain material itself, excluding void spaces.
        It can be calculated from bulk density and porosity using the formula:

        **True Density = Bulk Density / (1 - Porosity/100)**
        """)

        with st.form("true_density_form"):
            calculation_method = st.radio("Calculation Method", ["Use Previous Results", "Enter New Values"])

            if calculation_method == "Enter New Values":
                grain_type_true = st.text_input("Grain/Seed Type", "")
                bulk_density_input = st.number_input("Bulk Density (kg/m³)", min_value=0.1, value=750.0, step=0.1)
                porosity_input = st.number_input("Porosity (%)", min_value=0.0, max_value=99.9, value=40.0, step=0.1)
            else:
                st.info("This will use the average results from your previous calculations if available.")

            submit_button_true = st.form_submit_button("Calculate True Density")

        if submit_button_true:
            if calculation_method == "Enter New Values":
                grain_type_display = grain_type_true
                bulk_density_value = bulk_density_input
                porosity_value = porosity_input
            else:
                # Use values from previous calculations if available
                try:
                    grain_type_display = grain_type
                    bulk_density_value = avg_bulk_density * 1000  # Convert g/cc to kg/m³
                    porosity_value = avg_porosity
                except:
                    st.error("Previous calculation results not found. Please use 'Enter New Values' option.")
                    st.stop()

            # Calculate true density
            porosity_decimal = porosity_value / 100
            true_density = bulk_density_value / (1 - porosity_decimal)

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(f"### True Density Results for {grain_type_display}")

            data = {
                "Property": ["Bulk Density", "Porosity", "True Density"],
                "Value": [f"{bulk_density_value:.2f} kg/m³",
                          f"{porosity_value:.2f}%",
                          f"{true_density:.2f} kg/m³"]
            }

            results_df_true = pd.DataFrame(data)
            st.table(results_df_true)

            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization - comparison of bulk vs true density
            st.markdown("<h4>Visualization:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create a bar chart
            densities = [bulk_density_value, true_density]
            labels = ['Bulk Density', 'True Density']
            colors = ['skyblue', 'orange']

            ax.bar(labels, densities, color=colors, edgecolor='black')

            # Add values on top of bars
            for i, v in enumerate(densities):
                ax.text(i, v + 50, f"{v:.1f}", ha='center', fontsize=12)

            # Add text to illustrate porosity
            mid_point = (labels[0], (true_density + bulk_density_value) / 2)
            ax.annotate(f"Porosity: {porosity_value:.1f}%",
                        xy=(0.5, (true_density + bulk_density_value) / 2),
                        xytext=(1.0, (true_density + bulk_density_value) / 2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=12)

            ax.set_ylabel('Density (kg/m³)')
            ax.set_title(f'Comparison of Densities for {grain_type_display}')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Explanation of results
            st.markdown("""
            ### Interpretation of True Density Results

            True density represents the actual density of the grain material itself and:

            1. **Remains relatively constant** for a grain type regardless of packing
            2. **Varies with moisture content** - typically decreases as moisture increases
            3. **Provides insight into grain composition** - higher values may indicate higher starch content
            4. **Used in engineering calculations** for material properties

            Typical true density ranges:
            - Wheat: 1250-1450 kg/m³
            - Rice: 1300-1450 kg/m³
            - Corn: 1200-1350 kg/m³
            - Soybeans: 1150-1300 kg/m³
            """)

    # Comprehensive data table
    st.markdown("<h3 class='section-header'>Comprehensive Data Log</h3>", unsafe_allow_html=True)

    st.markdown("""
    Use this table to record and compare properties of different grain samples.
    You can manually enter data or use results from your calculations.
    """)

    # Create an editable dataframe
    if 'grain_data' not in st.session_state:
        st.session_state.grain_data = pd.DataFrame({
            'Grain/Seed': ['Wheat', 'Rice', 'Corn'],
            'Moisture Content (%)': [14.0, 12.0, 15.0],
            'Bulk Density (kg/m³)': [780, 570, 750],
            'Porosity (%)': [41.0, 48.0, 38.0],
            'True Density (kg/m³)': [1322, 1096, 1210]
        })

    # Create a form for adding new data
    with st.form("data_log_form"):
        st.markdown("#### Add New Data Row")

        col1, col2 = st.columns(2)
        with col1:
            new_grain = st.text_input("Grain/Seed Type", "")
            new_moisture = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0, value=14.0, step=0.1)

        with col2:
            new_bulk = st.number_input("Bulk Density (kg/m³)", min_value=0.1, value=750.0, step=0.1)
            new_porosity = st.number_input("Porosity (%)", min_value=0.1, max_value=99.9, value=40.0, step=0.1)

        # Calculate true density automatically
        new_true = new_bulk / (1 - new_porosity / 100)

        add_button = st.form_submit_button("Add to Data Log")

    if add_button and new_grain:
        # Add the new data to the dataframe
        new_row = pd.DataFrame({
            'Grain/Seed': [new_grain],
            'Moisture Content (%)': [new_moisture],
            'Bulk Density (kg/m³)': [new_bulk],
            'Porosity (%)': [new_porosity],
            'True Density (kg/m³)': [new_true]
        })

        st.session_state.grain_data = pd.concat([st.session_state.grain_data, new_row], ignore_index=True)
        st.success(f"Added {new_grain} to the data log!")

    # Display the current data
    st.dataframe(st.session_state.grain_data)

    # Add a button to download the data as CSV
    csv = st.session_state.grain_data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="grain_physical_properties.csv",
        mime="text/csv"
    )

    # Visualization of the comprehensive data
    if not st.session_state.grain_data.empty and len(st.session_state.grain_data) > 1:
        st.markdown("<h3 class='section-header'>Data Visualization</h3>", unsafe_allow_html=True)

        # Choose visualization type
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Bar Chart - All Properties", "Scatter Plot - Bulk vs True Density", "Bubble Chart - Density vs Moisture"]
        )

        if viz_type == "Bar Chart - All Properties":
            # Create grouped bar chart for all grains and their properties
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get the data
            grains = st.session_state.grain_data['Grain/Seed']
            bulk_data = st.session_state.grain_data['Bulk Density (kg/m³)']
            true_data = st.session_state.grain_data['True Density (kg/m³)']

            # Set positions and width for bars
            bar_width = 0.35
            r1 = np.arange(len(grains))
            r2 = [x + bar_width for x in r1]

            # Create bars
            ax.bar(r1, bulk_data, width=bar_width, label='Bulk Density', color='skyblue', edgecolor='navy')
            ax.bar(r2, true_data, width=bar_width, label='True Density', color='orange', edgecolor='darkred')

            # Add labels and legend
            ax.set_xlabel('Grain Type')
            ax.set_ylabel('Density (kg/m³)')
            ax.set_title('Comparison of Bulk and True Density Across Grain Types')
            ax.set_xticks([r + bar_width / 2 for r in range(len(grains))])
            ax.set_xticklabels(grains, rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

        elif viz_type == "Scatter Plot - Bulk vs True Density":
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot
            scatter = ax.scatter(
                st.session_state.grain_data['Bulk Density (kg/m³)'],
                st.session_state.grain_data['True Density (kg/m³)'],
                c=st.session_state.grain_data['Porosity (%)'],
                s=100,
                cmap='viridis',
                alpha=0.7
            )

            # Add labels for each point
            for i, txt in enumerate(st.session_state.grain_data['Grain/Seed']):
                ax.annotate(txt,
                            (st.session_state.grain_data['Bulk Density (kg/m³)'].iloc[i],
                             st.session_state.grain_data['True Density (kg/m³)'].iloc[i]),
                            xytext=(5, 5),
                            textcoords='offset points')

            # Add colorbar for porosity
            cbar = plt.colorbar(scatter)
            cbar.set_label('Porosity (%)')

            # Add diagonal line for reference (where porosity = 0)
            max_val = max(st.session_state.grain_data['True Density (kg/m³)'].max(),
                          st.session_state.grain_data['Bulk Density (kg/m³)'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Porosity = 0')

            ax.set_xlabel('Bulk Density (kg/m³)')
            ax.set_ylabel('True Density (kg/m³)')
            ax.set_title('Relationship Between Bulk and True Density')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

        elif viz_type == "Bubble Chart - Density vs Moisture":
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create bubble chart
            scatter = ax.scatter(
                st.session_state.grain_data['Moisture Content (%)'],
                st.session_state.grain_data['Bulk Density (kg/m³)'],
                s=st.session_state.grain_data['Porosity (%)'] * 10,  # Scale bubble size
                c=st.session_state.grain_data['True Density (kg/m³)'],
                cmap='plasma',
                alpha=0.7
            )

            # Add labels for each point
            for i, txt in enumerate(st.session_state.grain_data['Grain/Seed']):
                ax.annotate(txt,
                            (st.session_state.grain_data['Moisture Content (%)'].iloc[i],
                             st.session_state.grain_data['Bulk Density (kg/m³)'].iloc[i]),
                            xytext=(5, 5),
                            textcoords='offset points')

            # Add colorbar for true density
            cbar = plt.colorbar(scatter)
            cbar.set_label('True Density (kg/m³)')

            # Add legend for bubble size
            sizes = [20, 40, 60]
            labels = ['2%', '4%', '6%']
            legend_bubbles = []
            for size in sizes:
                legend_bubbles.append(ax.scatter([], [], s=size * 10, c='gray', alpha=0.7))

            ax.legend(legend_bubbles, labels, title='Porosity', loc='upper right', scatterpoints=1)

            ax.set_xlabel('Moisture Content (%)')
            ax.set_ylabel('Bulk Density (kg/m³)')
            ax.set_title('Relationship Between Moisture Content, Bulk Density, Porosity, and True Density')
            ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

# Grain Moisture Content Module
elif page == "Grain Moisture Content":
    st.markdown("<h2 class='sub-header'>Determination of Moisture Content of Various Grains</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This tool helps determine the moisture content of grain samples using direct methods.
    Moisture content is a critical parameter for grain quality, storage stability, and processing requirements.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Moisture Content Theory

        The amount of moisture in a grain or product is expressed as a percentage. This percentage can be expressed on either:

        1. **Wet Basis (w.b.)**: Moisture content relative to the wet weight of the sample (used commercially)
        2. **Dry Basis (d.b.)**: Moisture content relative to the dry weight of the sample (used in research)

        #### Key Definitions:

        - **Wet weight (W<sub>w</sub>)**: Weight of the sample before drying (g)
        - **Dry weight (W<sub>d</sub>)**: Weight of the sample after drying (g)
        - **Moisture weight (W<sub>m</sub>)**: Weight of moisture in the sample (W<sub>w</sub> - W<sub>d</sub>) (g)
        """, unsafe_allow_html=True)

        st.markdown("#### Calculation Formulas:")

        st.markdown("**Moisture Content on Wet Basis:**")
        st.latex(r"M_{wb} = \frac{W_m}{W_w} \times 100 = \frac{W_w - W_d}{W_w} \times 100")

        st.markdown("**Moisture Content on Dry Basis:**")
        st.latex(r"M_{db} = \frac{W_m}{W_d} \times 100 = \frac{W_w - W_d}{W_d} \times 100")

        st.markdown("#### Conversion Between Wet and Dry Basis:")

        st.markdown("**From Wet to Dry Basis:**")
        st.latex(r"M_{db} = \frac{M_{wb}}{100 - M_{wb}} \times 100")

        st.markdown("**From Dry to Wet Basis:**")
        st.latex(r"M_{wb} = \frac{M_{db}}{100 + M_{db}} \times 100")

        st.markdown("""
        #### Methods of Determination:

        **1. Direct Methods:**
        - **Oven Method**: Sample is dried in a hot air or vacuum oven under controlled temperature and time
        - **Distillation Method**: Using Dean-Stark apparatus or Brown Dual apparatus
        - **Infra-Red Meter Method**: Quick measurement using infrared radiation

        **2. Indirect Methods:**
        - Electrical resistance/conductance
        - Dielectric properties
        - Microwave absorption
        - Nuclear magnetic resonance

        #### Standard Drying Conditions for Different Materials:

        | Material | Standard | Temperature, °C | Duration |
        |----------|----------|----------------|----------|
        | Grains and nuts | ASCC | 130±1°C | 1 or 2 h |
        | Grains and nuts | ASCC | 100±1°C | 24 h |
        | Fruits and vegetables | ASAE | 70°C at 600 mm Hg | 6 h |
        | Spices | ASTA | 110±1°C - distillation | 1 to 3 h |

        The direct oven method is considered the standard reference method for moisture determination.
        """)
        # Show apparatus diagrams
        st.subheader("Apparatus for Moisture Determination:")
        col1, col2 = st.columns(2)

        with col1:
            st.image("https://via.placeholder.com/400x300.png?text=Dean-Stark+Apparatus",
                     caption="Fig 6.1: Dean-Stark Apparatus for Moisture Determination")

        with col2:
            st.image("https://via.placeholder.com/400x300.png?text=Brown+Dual+Apparatus",
                     caption="Fig 6.2: Brown Dual Apparatus for Moisture Determination")

        st.image("https://via.placeholder.com/800x300.png?text=Infra-Red+Moisture+Meter",
                 caption="Fig 6.3: Infra-Red Moisture Meter")

    # Create tabs for different operations
    tab1, tab2, tab3 = st.tabs(["Oven Method Calculator", "Moisture Content Converter", "Method Comparison"])

    with tab1:
        st.markdown("<h3 class='section-header'>Oven Method Moisture Content Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        The oven method is the standard reference method for determining moisture content. Enter your measurements below.
        """)

        # Input form for sample measurements
        with st.form("moisture_content_form"):
            grain_type = st.text_input("Grain/Seed Type", "")

            st.markdown("#### Weight Measurements:")

            col1, col2 = st.columns(2)
            with col1:
                empty_container = st.number_input("Empty Container Weight (g)", min_value=0.0, value=25.0, step=0.1)
                wet_container = st.number_input("Container + Wet Sample Weight (g)", min_value=0.0, value=35.0,
                                                step=0.1)
            with col2:
                dry_container = st.number_input("Container + Dry Sample Weight (g)", min_value=0.0, value=32.5,
                                                step=0.1)

            # Drying method selection
            drying_method = st.selectbox(
                "Drying Method",
                ["Hot Air Oven (130±1°C, 1-2h)", "Hot Air Oven (100±1°C, 24h)",
                 "Vacuum Oven (70°C, 6h)", "Custom Parameters"]
            )

            if drying_method == "Custom Parameters":
                col1, col2 = st.columns(2)
                with col1:
                    drying_temp = st.number_input("Drying Temperature (°C)", min_value=60.0, max_value=150.0,
                                                  value=105.0, step=5.0)
                with col2:
                    drying_time = st.number_input("Drying Time (hours)", min_value=0.5, max_value=48.0, value=3.0,
                                                  step=0.5)

            # Number of replications
            num_replications = st.number_input("Number of Replications", min_value=1, max_value=10, value=3, step=1)

            # Additional replications
            replication_data = []
            if num_replications > 1:
                st.markdown("#### Additional Replications:")
                for i in range(1, num_replications):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rep_empty = st.number_input(f"Empty Container - Rep {i + 1} (g)",
                                                    min_value=0.0, value=25.0, step=0.1, key=f"empty_{i}")
                    with col2:
                        rep_wet = st.number_input(f"Container + Wet Sample - Rep {i + 1} (g)",
                                                  min_value=0.0, value=35.0, step=0.1, key=f"wet_{i}")
                    with col3:
                        rep_dry = st.number_input(f"Container + Dry Sample - Rep {i + 1} (g)",
                                                  min_value=0.0, value=32.5, step=0.1, key=f"dry_{i}")

                    replication_data.append((rep_empty, rep_wet, rep_dry))

            calculate_button = st.form_submit_button("Calculate Moisture Content")

        if calculate_button:
            # Calculate weights for first measurement
            wet_weight = wet_container - empty_container
            dry_weight = dry_container - empty_container
            moisture_weight = wet_weight - dry_weight

            # Calculate moisture content
            moisture_wb = (moisture_weight / wet_weight) * 100  # wet basis
            moisture_db = (moisture_weight / dry_weight) * 100  # dry basis

            # Calculate for all replications
            all_moisture_wb = [moisture_wb]
            all_moisture_db = [moisture_db]
            all_wet_weights = [wet_weight]
            all_dry_weights = [dry_weight]
            all_moisture_weights = [moisture_weight]

            for empty, wet, dry in replication_data:
                rep_wet_weight = wet - empty
                rep_dry_weight = dry - empty
                rep_moisture_weight = rep_wet_weight - rep_dry_weight

                rep_moisture_wb = (rep_moisture_weight / rep_wet_weight) * 100
                rep_moisture_db = (rep_moisture_weight / rep_dry_weight) * 100

                all_wet_weights.append(rep_wet_weight)
                all_dry_weights.append(rep_dry_weight)
                all_moisture_weights.append(rep_moisture_weight)
                all_moisture_wb.append(rep_moisture_wb)
                all_moisture_db.append(rep_moisture_db)

            # Calculate averages
            avg_moisture_wb = sum(all_moisture_wb) / len(all_moisture_wb)
            avg_moisture_db = sum(all_moisture_db) / len(all_moisture_db)

            # Display method info
            if drying_method == "Hot Air Oven (130±1°C, 1-2h)":
                method_details = "Hot Air Oven at 130±1°C for 1-2 hours (ASCC Standard for Grains)"
            elif drying_method == "Hot Air Oven (100±1°C, 24h)":
                method_details = "Hot Air Oven at 100±1°C for 24 hours (ASCC Standard for Grains)"
            elif drying_method == "Vacuum Oven (70°C, 6h)":
                method_details = "Vacuum Oven at 70°C, 600 mm Hg for 6 hours (ASAE Standard)"
            else:
                method_details = f"Custom Parameters: {drying_temp}°C for {drying_time} hours"

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(f"### Moisture Content Results for {grain_type}")
            st.markdown(f"**Method Used:** {method_details}")

            # Create a table of results
            results_data = []
            for i in range(len(all_wet_weights)):
                rep_name = "Primary Sample" if i == 0 else f"Replication {i}"
                results_data.append({
                    "Sample": rep_name,
                    "Wet Weight (g)": round(all_wet_weights[i], 2),
                    "Dry Weight (g)": round(all_dry_weights[i], 2),
                    "Moisture Weight (g)": round(all_moisture_weights[i], 2),
                    "Moisture Content (% w.b.)": round(all_moisture_wb[i], 2),
                    "Moisture Content (% d.b.)": round(all_moisture_db[i], 2)
                })

            results_df = pd.DataFrame(results_data)
            st.table(results_df)

            st.markdown(f"""
            **Average Moisture Content:**
            - On Wet Basis (w.b.): {avg_moisture_wb:.2f}%
            - On Dry Basis (d.b.): {avg_moisture_db:.2f}%
            """)

            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization
            st.markdown("<h4>Visualization of Results:</h4>", unsafe_allow_html=True)

            # Create charts to visualize results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Sample indices
            samples = ["Primary"] + [f"Rep {i + 1}" for i in range(len(replication_data))]

            # Bar chart for moisture content
            x = np.arange(len(samples))
            width = 0.35

            ax1.bar(x - width / 2, all_moisture_wb, width, label='Wet Basis (%)', color='skyblue')
            ax1.bar(x + width / 2, all_moisture_db, width, label='Dry Basis (%)', color='salmon')

            # Add average lines
            ax1.axhline(y=avg_moisture_wb, color='blue', linestyle='--', alpha=0.7,
                        label=f'Avg. Wet Basis: {avg_moisture_wb:.2f}%')
            ax1.axhline(y=avg_moisture_db, color='red', linestyle='--', alpha=0.7,
                        label=f'Avg. Dry Basis: {avg_moisture_db:.2f}%')

            ax1.set_xlabel('Samples')
            ax1.set_ylabel('Moisture Content (%)')
            ax1.set_title('Moisture Content by Sample')
            ax1.set_xticks(x)
            ax1.set_xticklabels(samples)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Pie chart showing composition of primary sample
            primary_composition = [all_dry_weights[0], all_moisture_weights[0]]
            labels = [f'Dry Matter\n{100 - all_moisture_wb[0]:.1f}%', f'Moisture\n{all_moisture_wb[0]:.1f}%']
            colors = ['bisque', 'lightblue']

            ax2.pie(primary_composition, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
            ax2.axis('equal')
            ax2.set_title('Sample Composition (Primary Sample)')

            plt.tight_layout()
            st.pyplot(fig)

            # Explanation of results
            st.markdown("""
            ### Interpretation of Moisture Content Results

            Moisture content is a critical parameter for grain quality and storage stability:

            1. **Commercial Transactions**: Typically use wet basis (w.b.) measurements
            2. **Research and Calculations**: Typically use dry basis (d.b.) measurements
            3. **Storage Stability**: Higher moisture content increases risk of mold growth and spoilage
            4. **Processing**: Different moisture levels are optimal for different processing operations

            **Safe Storage Moisture Content (wet basis):**
            - Wheat: 12-14%
            - Rice: 12-14%
            - Corn: 13-15%
            - Soybeans: 11-13%

            For long-term storage, moisture content should generally be kept below these values to prevent spoilage.
            """)

    with tab2:
        st.markdown("<h3 class='section-header'>Moisture Content Converter</h3>", unsafe_allow_html=True)

        st.markdown("""
        This tool helps convert moisture content between wet basis (w.b.) and dry basis (d.b.).
        Enter a known moisture content value and convert it to the other basis.
        """)

        col1, col2 = st.columns(2)

        with col1:
            conversion_direction = st.radio("Conversion Direction",
                                            ["Wet Basis to Dry Basis", "Dry Basis to Wet Basis"])

            if conversion_direction == "Wet Basis to Dry Basis":
                moisture_input = st.number_input("Moisture Content (% w.b.)",
                                                 min_value=0.0, max_value=99.9, value=14.0, step=0.1)

                if st.button("Convert to Dry Basis"):
                    if moisture_input >= 100:
                        st.error("Moisture content on wet basis must be less than 100%")
                    else:
                        moisture_output = (moisture_input / (100 - moisture_input)) * 100

                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown(f"**{moisture_input:.2f}% (w.b.)** = **{moisture_output:.2f}% (d.b.)**")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                moisture_input = st.number_input("Moisture Content (% d.b.)",
                                                 min_value=0.0, value=16.3, step=0.1)

                if st.button("Convert to Wet Basis"):
                    moisture_output = (moisture_input / (100 + moisture_input)) * 100

                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"**{moisture_input:.2f}% (d.b.)** = **{moisture_output:.2f}% (w.b.)**")
                    st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### Conversion Formulas:")
            st.markdown("""
            **From Wet Basis to Dry Basis:**
            ```
            M(d.b.) = [M(w.b.) / (100 - M(w.b.))] × 100
            ```

            **From Dry Basis to Wet Basis:**
            ```
            M(w.b.) = [M(d.b.) / (100 + M(d.b.))] × 100
            ```

            Where:
            - M(w.b.) = Moisture content on wet basis (%)
            - M(d.b.) = Moisture content on dry basis (%)
            """)

            st.markdown("### Why Two Different Bases?")
            st.markdown("""
            - **Wet basis** is more intuitive and commonly used in commerce and trade
            - **Dry basis** is more useful in research and engineering calculations
            - Dry basis can exceed 100% for very moist materials
            - The conversion between them is important when comparing values from different sources
            """)

        # Visualization of the relationship between w.b. and d.b.
        st.markdown("<h4>Relationship Between Wet Basis and Dry Basis:</h4>", unsafe_allow_html=True)

        # Create data for the chart
        wb_values = np.arange(0, 95, 5)
        db_values = [(wb / (100 - wb)) * 100 for wb in wb_values]

        # Create DataFrame for the chart
        conversion_df = pd.DataFrame({
            'Moisture Content (% w.b.)': wb_values,
            'Moisture Content (% d.b.)': db_values
        })

        # Plot the relationship
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(wb_values, db_values, 'b-', linewidth=2)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)  # 1:1 line for reference

        # Add a point for the current conversion if button was pressed
        if 'moisture_input' in locals() and 'moisture_output' in locals():
            if conversion_direction == "Wet Basis to Dry Basis":
                ax.plot(moisture_input, moisture_output, 'ro', markersize=10)
                ax.annotate(f"({moisture_input:.1f}%, {moisture_output:.1f}%)",
                            (moisture_input, moisture_output),
                            xytext=(10, -15), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'))
            else:
                ax.plot(moisture_output, moisture_input, 'ro', markersize=10)
                ax.annotate(f"({moisture_output:.1f}%, {moisture_input:.1f}%)",
                            (moisture_output, moisture_input),
                            xytext=(10, -15), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel('Moisture Content (% w.b.)')
        ax.set_ylabel('Moisture Content (% d.b.)')
        ax.set_title('Conversion Between Wet Basis and Dry Basis')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set limits and annotations
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 400)

        # Add some annotations for context
        ax.annotate('Note: Dry basis can exceed 100%', xy=(70, 200), xytext=(70, 200),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

        plt.tight_layout()
        st.pyplot(fig)

        # Create a conversion table for reference
        st.markdown("<h4>Conversion Reference Table:</h4>", unsafe_allow_html=True)

        # Generate table data
        wb_table = list(range(5, 51, 5))
        db_table = [round((wb / (100 - wb)) * 100, 1) for wb in wb_table]

        # Create a DataFrame
        table_data = {
            'Wet Basis (%)': wb_table,
            'Dry Basis (%)': db_table
        }

        reference_df = pd.DataFrame(table_data)

        # Display the table
        st.table(reference_df)

    with tab3:
        st.markdown("<h3 class='section-header'>Moisture Measurement Methods Comparison</h3>", unsafe_allow_html=True)

        st.markdown("""
        Different methods are available for measuring grain moisture content, each with its own advantages and limitations.
        This comparison helps understand when to use each method.
        """)

        # Create comparison table
        method_data = {
            'Method': ['Hot Air Oven', 'Vacuum Oven', 'Distillation (Dean-Stark)', 'Infra-Red Moisture Meter',
                       'Electrical Moisture Meters'],
            'Accuracy': ['High', 'Very High', 'High', 'Medium', 'Medium-Low'],
            'Speed': ['Slow (1-24h)', 'Slow (6h+)', 'Medium (1-3h)', 'Fast (minutes)', 'Very Fast (seconds)'],
            'Sample Preparation': ['Simple', 'Simple', 'Complex', 'Simple', 'Simple'],
            'Equipment Cost': ['Low', 'High', 'Medium', 'Medium-High', 'Low-Medium'],
            'Best For': ['Reference method', 'Heat-sensitive materials', 'Oily/fatty materials', 'Rapid testing',
                         'Field testing'],
            'Limitations': ['Time-consuming', 'Complex equipment', 'Requires solvents', 'Calibration needed',
                            'Needs calibration for each grain type']
        }

        methods_df = pd.DataFrame(method_data)

        # Display the comparison table
        st.dataframe(methods_df)

        # Add visuals of different methods
        st.markdown("<h4>Visual Comparison of Methods:</h4>", unsafe_allow_html=True)

        # Display images in a grid
        col1, col2 = st.columns(2)

        with col1:
            st.image("https://via.placeholder.com/400x300.png?text=Hot+Air+Oven+Method",
                     caption="Hot Air Oven Method")

            st.image("https://via.placeholder.com/400x300.png?text=Dean-Stark+Distillation",
                     caption="Dean-Stark Distillation Method")

        with col2:
            st.image("https://via.placeholder.com/400x300.png?text=Infra-Red+Moisture+Meter",
                     caption="Infra-Red Moisture Meter Method")

            st.image("https://via.placeholder.com/400x300.png?text=Electrical+Moisture+Meters",
                     caption="Electrical Moisture Meters")

        # Add a decision flowchart
        st.markdown("<h4>Method Selection Flowchart:</h4>", unsafe_allow_html=True)

        st.image("https://via.placeholder.com/800x500.png?text=Moisture+Method+Selection+Flowchart",
                 caption="Decision Flowchart for Selecting a Moisture Measurement Method")

        # Interactive method selector
        st.markdown("<h4>Interactive Method Selector:</h4>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Selection Criteria")
            accuracy_needed = st.select_slider(
                "Accuracy Required",
                options=["Low", "Medium", "High", "Very High"],
                value="Medium"
            )

            time_available = st.select_slider(
                "Time Available",
                options=["Very Limited", "Limited", "Moderate", "Extensive"],
                value="Limited"
            )

            material_type = st.selectbox(
                "Material Type",
                ["Cereal Grains", "Oil Seeds", "Fruits/Vegetables", "Heat Sensitive Materials", "Oily/Fatty Materials"]
            )

            purpose = st.selectbox(
                "Purpose",
                ["Field Testing", "Quality Control", "Trade/Commerce", "Research", "Standard Reference"]
            )

        with col2:
            st.markdown("### Recommended Method")

            # Logic for method recommendation
            if purpose == "Standard Reference" or accuracy_needed == "Very High":
                recommended_method = "Vacuum Oven Method"
                reason = "Highest accuracy for reference measurements"
            elif material_type == "Oily/Fatty Materials":
                recommended_method = "Distillation Method (Dean-Stark)"
                reason = "Best for separating water from oils/fats"
            elif time_available == "Very Limited" and purpose == "Field Testing":
                recommended_method = "Electrical Moisture Meter"
                reason = "Fastest method for field use"
            elif time_available in ["Very Limited", "Limited"] and accuracy_needed in ["Medium", "High"]:
                recommended_method = "Infra-Red Moisture Meter"
                reason = "Good balance of speed and accuracy"
            else:
                recommended_method = "Hot Air Oven Method"
                reason = "Standard method with good accuracy"

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"### Recommended Method: {recommended_method}")
            st.markdown(f"**Reason:** {reason}")

            if recommended_method == "Hot Air Oven Method":
                st.markdown("""
                **Procedure:**
                1. Weigh empty container
                2. Add sample and weigh
                3. Dry at 130±1°C for 1-2 hours (grains)
                4. Cool in desiccator and weigh again
                5. Calculate moisture content
                """)
            elif recommended_method == "Vacuum Oven Method":
                st.markdown("""
                **Procedure:**
                1. Weigh empty container
                2. Add sample and weigh
                3. Dry at 70°C at 600 mm Hg for 6 hours
                4. Cool in desiccator and weigh again
                5. Calculate moisture content
                """)
            elif recommended_method == "Distillation Method (Dean-Stark)":
                st.markdown("""
                **Procedure:**
                1. Grind sample to appropriate size
                2. Place in Dean-Stark apparatus with toluene
                3. Heat to distill water and collect in graduated tube
                4. Read water volume directly from tube
                5. Calculate moisture content
                """)
            elif recommended_method == "Infra-Red Moisture Meter":
                st.markdown("""
                **Procedure:**
                1. Calibrate meter according to grain type
                2. Distribute sample evenly on sample pan
                3. Heat with infrared lamp until weight stabilizes
                4. Read moisture content directly from display
                """)
            else:  # Electrical Moisture Meter
                st.markdown("""
                **Procedure:**
                1. Select appropriate calibration for grain type
                2. Fill sample chamber with grain
                3. Press button to measure
                4. Read moisture content directly from display
                """)

            st.markdown("</div>", unsafe_allow_html=True)

    # Comprehensive data log for moisture content
    st.markdown("<h3 class='section-header'>Moisture Content Data Log</h3>", unsafe_allow_html=True)

    st.markdown("""
    Use this table to record and compare moisture content of different grain samples.
    You can manually enter data or use results from your calculations.
    """)

    # Create a session state for the data log if it doesn't exist
    if 'moisture_data' not in st.session_state:
        st.session_state.moisture_data = pd.DataFrame({
            'Grain/Seed': ['Wheat', 'Rice', 'Corn', 'Soybean'],
            'Measurement Method': ['Hot Air Oven', 'Hot Air Oven', 'Infra-Red', 'Vacuum Oven'],
            'Moisture Content (% w.b.)': [13.5, 12.0, 14.2, 10.8],
            'Moisture Content (% d.b.)': [15.6, 13.6, 16.6, 12.1],
            'Measurement Date': ['2023-01-15', '2023-01-15', '2023-01-16', '2023-01-16']
        })

    # Create a form for adding new data
    with st.form("moisture_log_form"):
        st.markdown("#### Add New Data Row")

        col1, col2 = st.columns(2)
        with col1:
            new_grain = st.text_input("Grain/Seed Type", "")
            new_method = st.selectbox(
                "Measurement Method",
                ["Hot Air Oven", "Vacuum Oven", "Distillation", "Infra-Red", "Electrical Meter"]
            )

        with col2:
            # Choose which basis to enter and calculate the other
            basis_choice = st.radio("Enter Moisture Content as:", ["Wet Basis (w.b.)", "Dry Basis (d.b.)"])

            if basis_choice == "Wet Basis (w.b.)":
                new_wb = st.number_input("Moisture Content (% w.b.)", min_value=0.0, max_value=99.9, value=14.0,
                                         step=0.1)
                # Calculate dry basis
                if new_wb < 100:
                    new_db = (new_wb / (100 - new_wb)) * 100
                else:
                    new_db = float('inf')  # Handle case where w.b. = 100%
            else:
                new_db = st.number_input("Moisture Content (% d.b.)", min_value=0.0, value=16.3, step=0.1)
                # Calculate wet basis
                new_wb = (new_db / (100 + new_db)) * 100

            # Current date as default
            new_date = st.date_input("Measurement Date")

        add_button = st.form_submit_button("Add to Data Log")

    if add_button and new_grain:
        # Add the new data to the dataframe
        new_row = pd.DataFrame({
            'Grain/Seed': [new_grain],
            'Measurement Method': [new_method],
            'Moisture Content (% w.b.)': [round(new_wb, 2)],
            'Moisture Content (% d.b.)': [round(new_db, 2)],
            'Measurement Date': [new_date.strftime("%Y-%m-%d")]
        })

        st.session_state.moisture_data = pd.concat([st.session_state.moisture_data, new_row], ignore_index=True)
        st.success(f"Added {new_grain} moisture data to the log!")

    # Display the current data
    st.dataframe(st.session_state.moisture_data)

    # Add a button to download the data as CSV
    csv = st.session_state.moisture_data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="grain_moisture_data.csv",
        mime="text/csv"
    )

    # Visualization of the comprehensive data
    if not st.session_state.moisture_data.empty and len(st.session_state.moisture_data) > 1:
        st.markdown("<h3 class='section-header'>Data Visualization</h3>", unsafe_allow_html=True)

        # Choose visualization type
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Bar Chart - Comparison by Grain", "Bar Chart - Comparison by Method", "Scatter Plot - W.B. vs D.B."]
        )

        if viz_type == "Bar Chart - Comparison by Grain":
            # Create bar chart comparing moisture content across grains
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get the data
            grain_data = st.session_state.moisture_data.sort_values('Grain/Seed')
            grains = grain_data['Grain/Seed']
            wb_data = grain_data['Moisture Content (% w.b.)']
            db_data = grain_data['Moisture Content (% d.b.)']
            methods = grain_data['Measurement Method']

            # Create bar positions
            x = np.arange(len(grains))
            width = 0.35

            # Create bars with custom colors based on method
            method_colors_wb = {
                'Hot Air Oven': 'skyblue',
                'Vacuum Oven': 'royalblue',
                'Distillation': 'lightgreen',
                'Infra-Red': 'salmon',
                'Electrical Meter': 'plum'
            }

            method_colors_db = {
                'Hot Air Oven': 'blue',
                'Vacuum Oven': 'darkblue',
                'Distillation': 'green',
                'Infra-Red': 'red',
                'Electrical Meter': 'purple'
            }

            wb_bars = ax.bar(x - width / 2, wb_data, width, label='Wet Basis (%)',
                             color=[method_colors_wb.get(m, 'gray') for m in methods])

            db_bars = ax.bar(x + width / 2, db_data, width, label='Dry Basis (%)',
                             color=[method_colors_db.get(m, 'darkgray') for m in methods])

            # Add labels and legend
            ax.set_xlabel('Grain Type')
            ax.set_ylabel('Moisture Content (%)')
            ax.set_title('Comparison of Moisture Content by Grain Type')
            ax.set_xticks(x)
            ax.set_xticklabels(grains, rotation=45, ha='right')

            # Add a legend for the basis
            basis_legend = ax.legend(loc='upper left')

            # Add a second legend for the methods
            method_patches = [plt.Rectangle((0, 0), 1, 1, color=method_colors_wb[m]) for m in method_colors_wb]
            ax.legend(method_patches, method_colors_wb.keys(), loc='upper right', title="Method")
            ax.add_artist(basis_legend)  # Add the first legend back

            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

        elif viz_type == "Bar Chart - Comparison by Method":
            # Create bar chart comparing moisture content across methods
            fig, ax = plt.subplots(figsize=(12, 6))

            # Group data by method and calculate means
            method_data = st.session_state.moisture_data.groupby('Measurement Method').agg({
                'Moisture Content (% w.b.)': 'mean',
                'Moisture Content (% d.b.)': 'mean'
            }).reset_index()

            # Sort by wet basis moisture content
            method_data = method_data.sort_values('Moisture Content (% w.b.)')

            methods = method_data['Measurement Method']
            wb_means = method_data['Moisture Content (% w.b.)']
            db_means = method_data['Moisture Content (% d.b.)']

            # Create bar positions
            x = np.arange(len(methods))
            width = 0.35

            # Create bars
            ax.bar(x - width / 2, wb_means, width, label='Wet Basis (%)', color='skyblue')
            ax.bar(x + width / 2, db_means, width, label='Dry Basis (%)', color='salmon')

            # Add labels and legend
            ax.set_xlabel('Measurement Method')
            ax.set_ylabel('Average Moisture Content (%)')
            ax.set_title('Average Moisture Content by Measurement Method')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()

            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

        elif viz_type == "Scatter Plot - W.B. vs D.B.":
            # Create scatter plot of wet basis vs dry basis values
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get the data
            scatter_data = st.session_state.moisture_data
            wb_values = scatter_data['Moisture Content (% w.b.)']
            db_values = scatter_data['Moisture Content (% d.b.)']
            grains = scatter_data['Grain/Seed']
            methods = scatter_data['Measurement Method']

            # Create a colormap for different methods
            method_types = methods.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(method_types)))
            color_map = dict(zip(method_types, colors))

            # Create scatter plot with different colors for methods
            for method in method_types:
                mask = methods == method
                ax.scatter(
                    wb_values[mask], db_values[mask],
                    label=method,
                    s=100,
                    color=color_map[method],
                    alpha=0.7,
                    edgecolor='black'
                )

            # Add labels for each point
            for i, txt in enumerate(grains):
                ax.annotate(txt,
                            (wb_values.iloc[i], db_values.iloc[i]),
                            xytext=(5, 5), textcoords='offset points')

            # Add the theoretical relationship line
            x_line = np.linspace(0, max(wb_values) * 1.1, 100)
            y_line = [(x / (100 - x)) * 100 for x in x_line]
            ax.plot(x_line, y_line, 'k--', alpha=0.5, label='Theoretical Relationship')

            # Set labels and legend
            ax.set_xlabel('Moisture Content (% w.b.)')
            ax.set_ylabel('Moisture Content (% d.b.)')
            ax.set_title('Relationship Between Wet Basis and Dry Basis Measurements')
            ax.legend()

            ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

# Terminal Velocity Module
elif page == "Terminal Velocity":
    st.markdown("<h2 class='sub-header'>Terminal Velocity of Grains</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This tool helps determine the terminal velocity of grain samples, which is a critical aerodynamic property 
    for designing grain handling systems, pneumatic transport, and separation equipment.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Terminal Velocity Theory

        Terminal velocity is the constant velocity achieved by a falling object when the force of air resistance 
        equals the force of gravity. For grains, it represents the air velocity at which the grain is kept in 
        suspension against gravity.

        #### Importance of Terminal Velocity:

        1. **Pneumatic Separation**: Used to separate desirable products from unwanted materials
        2. **Material Transport**: Essential for air conveying systems design
        3. **Equipment Design**: Needed for sizing and designing:
           - Aspirators
           - Air classifiers
           - Pneumatic conveying systems
           - Cleaning equipment

        #### Factors Affecting Terminal Velocity:

        - **Physical Properties**: Size, shape, density, and surface characteristics of the grain
        - **Orientation**: How the particle orients itself in an air stream
        - **Moisture Content**: Higher moisture typically increases terminal velocity
        - **Air Properties**: Density and viscosity of the air

        #### Measurement Method:

        Terminal velocity is typically measured using a vertical air column apparatus where:
        1. Grains are placed in a vertical transparent tube
        2. Air is blown upward through the tube
        3. The air velocity is gradually increased until the grain is suspended
        4. The velocity at which the grain remains suspended (neither rising nor falling) is the terminal velocity

        This velocity (measured in m/s) represents the minimum airflow needed to keep the grain in suspension.
        """)

        # Show terminal velocity apparatus diagram
        st.image("https://via.placeholder.com/800x400.png?text=Terminal+Velocity+Apparatus",
                 caption="Fig 8.1: Apparatus for determination of terminal velocity of grains")

    # Create tabs for different operations
    tab1, tab2, tab3 = st.tabs(
        ["Terminal Velocity Calculator", "Comparative Analysis", "Factors Affecting Terminal Velocity"])

    with tab1:
        st.markdown("<h3 class='section-header'>Terminal Velocity Measurement</h3>", unsafe_allow_html=True)

        st.markdown("""
        Enter the measurements from your terminal velocity experiment. The terminal velocity is the air velocity
        at which the grain particles remain suspended in the air stream, neither rising nor falling.
        """)

        # Input form for sample measurements
        with st.form("terminal_velocity_form"):
            grain_type = st.text_input("Grain/Seed Type (Variety)", "")
            moisture_content = st.number_input("Moisture Content (% d.b.)", min_value=0.0, max_value=100.0, value=14.0,
                                               step=0.1)

            # Terminal velocity measurements
            st.markdown("#### Velocity Measurements:")

            col1, col2 = st.columns(2)
            with col1:
                # Add option to select measurement units
                velocity_units = st.selectbox("Velocity Units", ["m/s", "ft/s", "km/h", "mph"])
                velocity_value = st.number_input(f"Terminal Velocity ({velocity_units})", min_value=0.0, value=8.0,
                                                 step=0.1)

            with col2:
                air_temperature = st.number_input("Air Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0,
                                                  step=0.5)
                air_pressure = st.number_input("Atmospheric Pressure (kPa)", min_value=80.0, max_value=110.0,
                                               value=101.3, step=0.1)

            # Number of replications
            num_replications = st.number_input("Number of Replications", min_value=1, max_value=10, value=3, step=1)

            # Additional replications
            replication_data = []
            if num_replications > 1:
                st.markdown("#### Additional Replications:")
                for i in range(1, num_replications):
                    rep_velocity = st.number_input(f"Terminal Velocity ({velocity_units}) - Rep {i + 1}",
                                                   min_value=0.0, value=8.0, step=0.1, key=f"vel_{i}")
                    replication_data.append(rep_velocity)

            calculate_button = st.form_submit_button("Calculate Statistics")

        if calculate_button:
            # Convert velocity to m/s if needed
            if velocity_units == "ft/s":
                velocity_ms = velocity_value * 0.3048
            elif velocity_units == "km/h":
                velocity_ms = velocity_value / 3.6
            elif velocity_units == "mph":
                velocity_ms = velocity_value * 0.44704
            else:  # m/s
                velocity_ms = velocity_value

            # Convert replications to m/s
            all_velocities_ms = [velocity_ms]
            for rep_velocity in replication_data:
                if velocity_units == "ft/s":
                    rep_velocity_ms = rep_velocity * 0.3048
                elif velocity_units == "km/h":
                    rep_velocity_ms = rep_velocity / 3.6
                elif velocity_units == "mph":
                    rep_velocity_ms = rep_velocity * 0.44704
                else:  # m/s
                    rep_velocity_ms = rep_velocity
                all_velocities_ms.append(rep_velocity_ms)

            # Calculate statistics
            avg_velocity_ms = sum(all_velocities_ms) / len(all_velocities_ms)
            min_velocity_ms = min(all_velocities_ms)
            max_velocity_ms = max(all_velocities_ms)
            std_velocity_ms = np.std(all_velocities_ms)

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown(f"### Terminal Velocity Results for {grain_type}")
            st.markdown(f"**Moisture Content:** {moisture_content:.1f}% (d.b.)")

            # Create a table of results
            results_data = []
            for i in range(len(all_velocities_ms)):
                rep_name = "Primary Measurement" if i == 0 else f"Replication {i}"

                # Convert back to original units for display
                if velocity_units == "ft/s":
                    display_value = all_velocities_ms[i] / 0.3048
                elif velocity_units == "km/h":
                    display_value = all_velocities_ms[i] * 3.6
                elif velocity_units == "mph":
                    display_value = all_velocities_ms[i] / 0.44704
                else:  # m/s
                    display_value = all_velocities_ms[i]

                results_data.append({
                    "Measurement": rep_name,
                    f"Terminal Velocity ({velocity_units})": round(display_value, 2),
                    "Terminal Velocity (m/s)": round(all_velocities_ms[i], 2)
                })

            results_df = pd.DataFrame(results_data)
            st.table(results_df)

            # Display statistics
            st.markdown("### Statistical Summary:")

            # Convert statistics back to selected units for display
            if velocity_units == "ft/s":
                avg_velocity_display = avg_velocity_ms / 0.3048
                min_velocity_display = min_velocity_ms / 0.3048
                max_velocity_display = max_velocity_ms / 0.3048
                std_velocity_display = std_velocity_ms / 0.3048
            elif velocity_units == "km/h":
                avg_velocity_display = avg_velocity_ms * 3.6
                min_velocity_display = min_velocity_ms * 3.6
                max_velocity_display = max_velocity_ms * 3.6
                std_velocity_display = std_velocity_ms * 3.6
            elif velocity_units == "mph":
                avg_velocity_display = avg_velocity_ms / 0.44704
                min_velocity_display = min_velocity_ms / 0.44704
                max_velocity_display = max_velocity_ms / 0.44704
                std_velocity_display = std_velocity_ms / 0.44704
            else:  # m/s
                avg_velocity_display = avg_velocity_ms
                min_velocity_display = min_velocity_ms
                max_velocity_display = max_velocity_ms
                std_velocity_display = std_velocity_ms

            stats_data = {
                "Statistic": ["Average", "Minimum", "Maximum", "Standard Deviation"],
                f"Value ({velocity_units})": [
                    round(avg_velocity_display, 2),
                    round(min_velocity_display, 2),
                    round(max_velocity_display, 2),
                    round(std_velocity_display, 2)
                ],
                "Value (m/s)": [
                    round(avg_velocity_ms, 2),
                    round(min_velocity_ms, 2),
                    round(max_velocity_ms, 2),
                    round(std_velocity_ms, 2)
                ]
            }

            stats_df = pd.DataFrame(stats_data)
            st.table(stats_df)

            st.markdown("</div>", unsafe_allow_html=True)

            # Visualization
            if len(all_velocities_ms) > 1:
                st.markdown("<h4>Visualization of Measurements:</h4>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 6))

                # Create bar chart with original units
                if velocity_units == "ft/s":
                    display_values = [v / 0.3048 for v in all_velocities_ms]
                elif velocity_units == "km/h":
                    display_values = [v * 3.6 for v in all_velocities_ms]
                elif velocity_units == "mph":
                    display_values = [v / 0.44704 for v in all_velocities_ms]
                else:  # m/s
                    display_values = all_velocities_ms

                reps = ["Primary"] + [f"Rep {i + 1}" for i in range(len(replication_data))]
                ax.bar(reps, display_values, color='lightblue', edgecolor='navy')

                # Add a horizontal line for the average
                ax.axhline(y=avg_velocity_display, color='red', linestyle='--',
                           label=f'Average: {avg_velocity_display:.2f} {velocity_units}')

                ax.set_ylabel(f'Terminal Velocity ({velocity_units})')
                ax.set_title(f'Terminal Velocity Measurements for {grain_type}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

            # Explanation of results
            st.markdown("""
            ### Interpretation of Terminal Velocity Results

            Terminal velocity is a critical parameter for:

            1. **Pneumatic Conveying**: Design of air transport systems
            2. **Grain Cleaning**: Separation of grains from lighter impurities
            3. **Classifier Design**: Creating equipment to sort grains by size/density
            4. **Drying Systems**: Ensuring grains remain in drying chambers

            Typical terminal velocity ranges:
            - Wheat: 8-11 m/s
            - Rice: 7-10 m/s
            - Corn: 10-14 m/s
            - Soybeans: 9-13 m/s
            - Millet: 5-8 m/s

            *Note: Terminal velocity increases with moisture content and particle size.*
            """)

    with tab2:
        st.markdown("<h3 class='section-header'>Comparative Analysis</h3>", unsafe_allow_html=True)

        st.markdown("""
        Compare terminal velocities of different grain types and analyze how they vary with moisture content.
        This analysis is useful for designing grain handling and separation systems that process multiple grain types.
        """)

        # Create a data log for terminal velocity measurements
        if 'terminal_velocity_data' not in st.session_state:
            st.session_state.terminal_velocity_data = pd.DataFrame({
                'Grain/Seed': ['Wheat', 'Rice', 'Corn', 'Soybean', 'Millet'],
                'Moisture Content (% d.b.)': [14.0, 12.0, 15.0, 12.5, 11.0],
                'Terminal Velocity (m/s)': [9.5, 8.2, 12.3, 10.8, 6.5],
                'Density (kg/m³)': [1250, 1150, 1300, 1180, 1100],
                'Size (mm)': [4.0, 7.0, 10.0, 7.5, 2.5]
            })

        # Create a form for adding new data
        with st.form("terminal_velocity_log_form"):
            st.markdown("#### Add New Data Row")

            col1, col2, col3 = st.columns(3)
            with col1:
                new_grain = st.text_input("Grain/Seed Type", "")
                new_moisture = st.number_input("Moisture Content (% d.b.)", min_value=0.0, max_value=100.0, value=14.0,
                                               step=0.1)

            with col2:
                new_velocity = st.number_input("Terminal Velocity (m/s)", min_value=0.0, value=8.0, step=0.1)
                new_density = st.number_input("Density (kg/m³)", min_value=0.0, value=1200.0, step=10.0)

            with col3:
                new_size = st.number_input("Size (mm)", min_value=0.0, value=5.0, step=0.1)
                # Option to add from previous calculation
                use_previous = st.checkbox("Use data from previous calculation")

            add_button = st.form_submit_button("Add to Data Log")

        if add_button and new_grain:
            # Add the new data to the dataframe
            new_row = pd.DataFrame({
                'Grain/Seed': [new_grain],
                'Moisture Content (% d.b.)': [new_moisture],
                'Terminal Velocity (m/s)': [new_velocity],
                'Density (kg/m³)': [new_density],
                'Size (mm)': [new_size]
            })

            st.session_state.terminal_velocity_data = pd.concat([st.session_state.terminal_velocity_data, new_row],
                                                                ignore_index=True)
            st.success(f"Added {new_grain} to the data log!")

        # Display the current data
        st.dataframe(st.session_state.terminal_velocity_data)

        # Add a button to download the data as CSV
        csv = st.session_state.terminal_velocity_data.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="grain_terminal_velocity_data.csv",
            mime="text/csv"
        )

        # Create visualizations
        if not st.session_state.terminal_velocity_data.empty and len(st.session_state.terminal_velocity_data) > 1:
            st.markdown("<h4>Visualization:</h4>", unsafe_allow_html=True)

            viz_type = st.selectbox(
                "Choose Visualization Type",
                ["Bar Chart - Terminal Velocity by Grain Type",
                 "Scatter Plot - Terminal Velocity vs Moisture Content",
                 "Scatter Plot - Terminal Velocity vs Size",
                 "Scatter Plot - Terminal Velocity vs Density"]
            )

            if viz_type == "Bar Chart - Terminal Velocity by Grain Type":
                # Sort data by terminal velocity
                sorted_data = st.session_state.terminal_velocity_data.sort_values('Terminal Velocity (m/s)')

                fig, ax = plt.subplots(figsize=(12, 6))

                # Create bar chart with error bar for standard deviation
                ax.bar(sorted_data['Grain/Seed'], sorted_data['Terminal Velocity (m/s)'],
                       color='skyblue', edgecolor='navy')

                ax.set_xlabel('Grain Type')
                ax.set_ylabel('Terminal Velocity (m/s)')
                ax.set_title('Comparison of Terminal Velocities by Grain Type')
                plt.xticks(rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

            elif viz_type == "Scatter Plot - Terminal Velocity vs Moisture Content":
                fig, ax = plt.subplots(figsize=(10, 6))

                # Create scatter plot
                for grain in st.session_state.terminal_velocity_data['Grain/Seed'].unique():
                    grain_data = st.session_state.terminal_velocity_data[
                        st.session_state.terminal_velocity_data['Grain/Seed'] == grain]
                    ax.scatter(grain_data['Moisture Content (% d.b.)'], grain_data['Terminal Velocity (m/s)'],
                               label=grain, s=80, alpha=0.7)

                # Add regression line if enough data points
                if len(st.session_state.terminal_velocity_data) > 2:
                    x = st.session_state.terminal_velocity_data['Moisture Content (% d.b.)']
                    y = st.session_state.terminal_velocity_data['Terminal Velocity (m/s)']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "r--", alpha=0.7, label="Trend line")

                ax.set_xlabel('Moisture Content (% d.b.)')
                ax.set_ylabel('Terminal Velocity (m/s)')
                ax.set_title('Relationship Between Moisture Content and Terminal Velocity')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                **Note:** Terminal velocity typically increases with moisture content due to increased 
                particle density and changes in aerodynamic properties. The relationship is often linear 
                within common moisture content ranges.
                """)

            elif viz_type == "Scatter Plot - Terminal Velocity vs Size":
                fig, ax = plt.subplots(figsize=(10, 6))

                # Create scatter plot
                scatter = ax.scatter(
                    st.session_state.terminal_velocity_data['Size (mm)'],
                    st.session_state.terminal_velocity_data['Terminal Velocity (m/s)'],
                    c=st.session_state.terminal_velocity_data['Density (kg/m³)'],
                    s=80,
                    alpha=0.7,
                    cmap='viridis'
                )

                # Add labels for each point
                for i, txt in enumerate(st.session_state.terminal_velocity_data['Grain/Seed']):
                    ax.annotate(txt,
                                (st.session_state.terminal_velocity_data['Size (mm)'].iloc[i],
                                 st.session_state.terminal_velocity_data['Terminal Velocity (m/s)'].iloc[i]),
                                xytext=(5, 5),
                                textcoords='offset points')

                # Add regression line if enough data points
                if len(st.session_state.terminal_velocity_data) > 2:
                    x = st.session_state.terminal_velocity_data['Size (mm)']
                    y = st.session_state.terminal_velocity_data['Terminal Velocity (m/s)']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "r--", alpha=0.7, label="Trend line")

                # Add a color bar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Density (kg/m³)')

                ax.set_xlabel('Size (mm)')
                ax.set_ylabel('Terminal Velocity (m/s)')
                ax.set_title('Relationship Between Grain Size and Terminal Velocity')
                ax.grid(True, linestyle='--', alpha=0.7)
                if len(st.session_state.terminal_velocity_data) > 2:
                    ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                **Note:** Generally, terminal velocity increases with particle size due to the 
                increased weight-to-drag ratio. However, shape factors can also influence this relationship.
                """)

            elif viz_type == "Scatter Plot - Terminal Velocity vs Density":
                fig, ax = plt.subplots(figsize=(10, 6))

                # Create scatter plot
                scatter = ax.scatter(
                    st.session_state.terminal_velocity_data['Density (kg/m³)'],
                    st.session_state.terminal_velocity_data['Terminal Velocity (m/s)'],
                    c=st.session_state.terminal_velocity_data['Size (mm)'],
                    s=80,
                    alpha=0.7,
                    cmap='plasma'
                )

                # Add labels for each point
                for i, txt in enumerate(st.session_state.terminal_velocity_data['Grain/Seed']):
                    ax.annotate(txt,
                                (st.session_state.terminal_velocity_data['Density (kg/m³)'].iloc[i],
                                 st.session_state.terminal_velocity_data['Terminal Velocity (m/s)'].iloc[i]),
                                xytext=(5, 5),
                                textcoords='offset points')

                # Add regression line if enough data points
                if len(st.session_state.terminal_velocity_data) > 2:
                    x = st.session_state.terminal_velocity_data['Density (kg/m³)']
                    y = st.session_state.terminal_velocity_data['Terminal Velocity (m/s)']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "r--", alpha=0.7, label="Trend line")

                # Add a color bar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Size (mm)')

                ax.set_xlabel('Density (kg/m³)')
                ax.set_ylabel('Terminal Velocity (m/s)')
                ax.set_title('Relationship Between Grain Density and Terminal Velocity')
                ax.grid(True, linestyle='--', alpha=0.7)
                if len(st.session_state.terminal_velocity_data) > 2:
                    ax.legend()

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                **Note:** Terminal velocity typically increases with particle density. 
                Denser particles require higher air velocities to remain suspended.
                """)

    with tab3:
        st.markdown("<h3 class='section-header'>Factors Affecting Terminal Velocity</h3>", unsafe_allow_html=True)

        st.markdown("""
        This interactive simulator demonstrates how different factors affect the terminal velocity of grains.
        Adjust the parameters to see how they influence terminal velocity.
        """)

        # Create a simple terminal velocity prediction model
        st.markdown("### Terminal Velocity Simulator")

        # Create a form for the simulator
        col1, col2 = st.columns(2)

        with col1:
            sim_density = st.slider("Grain Density (kg/m³)", 800, 1600, 1200, 50)
            sim_diameter = st.slider("Equivalent Diameter (mm)", 1.0, 15.0, 5.0, 0.5)
            sim_shape_factor = st.slider("Shape Factor (0.5-1.0)", 0.5, 1.0, 0.8, 0.05)

        with col2:
            sim_air_density = st.slider("Air Density (kg/m³)", 1.0, 1.4, 1.2, 0.05)
            sim_air_viscosity = st.slider("Air Viscosity (×10⁻⁵ Pa·s)", 1.5, 2.5, 1.8, 0.1) * 1e-5
            sim_drag_coefficient = st.slider("Drag Coefficient", 0.4, 1.0, 0.6, 0.05)

        # Simple terminal velocity equation (simplified for demonstration)
        # Terminal velocity = sqrt((4/3) * (g * d * ρp * shape_factor) / (CD * ρa))
        # where g = gravity, d = diameter, ρp = particle density, ρa = air density, CD = drag coefficient

        g = 9.81  # m/s²
        d = sim_diameter / 1000  # Convert mm to m

        terminal_velocity = np.sqrt((4 / 3) * (g * d * sim_density * sim_shape_factor) /
                                    (sim_drag_coefficient * sim_air_density))

        st.markdown(f"""
        <div class='result-box'>
        <h3>Predicted Terminal Velocity: {terminal_velocity:.2f} m/s</h3>
        </div>
        """, unsafe_allow_html=True)

        # Create a radar chart to show the sensitivity of each parameter
        st.markdown("### Parameter Sensitivity Analysis")

        # Calculate how much each parameter affects the terminal velocity
        base_velocity = terminal_velocity

        # Increase each parameter by 10% and see how it changes terminal velocity
        density_sensitivity = (np.sqrt((4 / 3) * (g * d * (sim_density * 1.1) * sim_shape_factor) /
                                       (sim_drag_coefficient * sim_air_density)) - base_velocity) / base_velocity * 100

        diameter_sensitivity = (np.sqrt((4 / 3) * (g * (d * 1.1) * sim_density * sim_shape_factor) /
                                        (sim_drag_coefficient * sim_air_density)) - base_velocity) / base_velocity * 100

        shape_sensitivity = (np.sqrt((4 / 3) * (g * d * sim_density * (sim_shape_factor * 1.1)) /
                                     (sim_drag_coefficient * sim_air_density)) - base_velocity) / base_velocity * 100

        air_density_sensitivity = (np.sqrt((4 / 3) * (g * d * sim_density * sim_shape_factor) /
                                           (sim_drag_coefficient * (
                                                       sim_air_density * 1.1))) - base_velocity) / base_velocity * 100

        drag_sensitivity = (np.sqrt((4 / 3) * (g * d * sim_density * sim_shape_factor) /
                                    ((
                                                 sim_drag_coefficient * 1.1) * sim_air_density)) - base_velocity) / base_velocity * 100

        # Create bar chart for sensitivity analysis
        sensitivity_data = {
            'Parameter': ['Grain Density', 'Grain Diameter', 'Shape Factor', 'Air Density', 'Drag Coefficient'],
            'Sensitivity (% change)': [
                density_sensitivity,
                diameter_sensitivity,
                shape_sensitivity,
                air_density_sensitivity,
                drag_sensitivity
            ]
        }

        sensitivity_df = pd.DataFrame(sensitivity_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use different colors based on whether the effect is positive or negative
        colors = ['green' if x > 0 else 'red' for x in sensitivity_df['Sensitivity (% change)']]

        ax.bar(sensitivity_df['Parameter'], sensitivity_df['Sensitivity (% change)'], color=colors)

        ax.set_xlabel('Parameter')
        ax.set_ylabel('% Change in Terminal Velocity for 10% Parameter Increase')
        ax.set_title('Sensitivity Analysis of Terminal Velocity Parameters')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        ### Theoretical Relationships

        The terminal velocity of a particle depends on several factors according to the following simplified model:

        **Terminal Velocity (V<sub>t</sub>) = √[(4/3) × (g × d × ρ<sub>p</sub> × SF) / (C<sub>D</sub> × ρ<sub>a</sub>)]**

        Where:
        - g = Gravitational acceleration (9.81 m/s²)
        - d = Particle diameter (m)
        - ρ<sub>p</sub> = Particle density (kg/m³)
        - SF = Shape factor (dimensionless)
        - C<sub>D</sub> = Drag coefficient (dimensionless)
        - ρ<sub>a</sub> = Air density (kg/m³)

        #### Key Relationships:

        1. **Particle Size**: Terminal velocity increases with the square root of diameter
        2. **Particle Density**: Terminal velocity increases with the square root of density
        3. **Air Density**: Terminal velocity decreases with the square root of air density
        4. **Shape Factor**: More spherical particles (higher SF) have higher terminal velocities
        5. **Drag Coefficient**: Higher drag reduces terminal velocity

        These relationships explain why larger, denser grains have higher terminal velocities,
        and why terminal velocity varies with moisture content (which affects density) and with
        different grain shapes.
        """, unsafe_allow_html=True)

    # Comprehensive data table
    st.markdown("<h3 class='section-header'>Typical Terminal Velocities Reference</h3>", unsafe_allow_html=True)

    st.markdown("""
    The following table provides reference values for terminal velocities of common agricultural materials.
    These values can vary based on variety, moisture content, and physical properties.
    """)

    # Create a reference table
    reference_data = {
        'Material': [
            'Wheat', 'Rice (paddy)', 'Rice (brown)', 'Corn', 'Soybean',
            'Barley', 'Sorghum', 'Millet', 'Oats', 'Lentil',
            'Chickpea', 'Sunflower seed', 'Rapeseed', 'Cotton seed', 'Peanut'
        ],
        'Terminal Velocity Range (m/s)': [
            '8.0 - 11.0', '7.0 - 9.5', '6.8 - 9.0', '10.0 - 14.0', '9.0 - 13.0',
            '8.5 - 11.5', '8.0 - 11.0', '5.0 - 8.0', '6.5 - 9.5', '7.0 - 10.0',
            '9.5 - 13.0', '5.0 - 7.5', '4.0 - 7.0', '6.5 - 9.0', '11.0 - 16.0'
        ],
        'Typical Moisture Content (% d.b.)': [
            '12 - 16', '14 - 18', '12 - 14', '14 - 18', '12 - 16',
            '12 - 16', '12 - 15', '10 - 14', '12 - 15', '10 - 14',
            '10 - 14', '8 - 12', '8 - 10', '10 - 14', '8 - 12'
        ],
        'Applications': [
            'Cleaning, pneumatic conveying', 'Husk separation, cleaning', 'Cleaning, grading',
            'Cleaning, drying, transport', 'Cleaning, separation',
            'Husk removal, cleaning', 'Cleaning, separation', 'Cleaning, impurity removal',
            'Dehulling, cleaning', 'Separation, cleaning',
            'Separation, cleaning', 'Dehulling, cleaning', 'Cleaning, separation',
            'Lint removal, cleaning', 'Separation, cleaning'
        ]
    }

    reference_df = pd.DataFrame(reference_data)
    st.dataframe(reference_df)

    st.markdown("""
    ### Applications of Terminal Velocity Knowledge

    Understanding terminal velocities allows for the design of:

    1. **Pneumatic Conveying Systems**: Designing air transport systems with appropriate velocities
    2. **Air Classifiers**: Separating materials based on aerodynamic properties
    3. **Cleaning Equipment**: Removing lighter impurities from grain
    4. **Drying Systems**: Ensuring grains remain in suspension for proper drying
    5. **Storage Handling**: Preventing excessive grain damage during pneumatic handling

    The terminal velocity of a material provides a critical design parameter that determines
    the minimum air velocity needed for suspension, and the maximum velocity that prevents
    unwanted material loss during separation operations.
    """)

# Screen Cleaner Evaluation Module
elif page == "Screen Cleaner Evaluation":
    st.markdown("<h2 class='sub-header'>Performance Evaluation of Screen Cleaner/Grader</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This module helps determine the effectiveness of grain cleaners and graders by evaluating the efficiency 
    of cleaning and separation processes. Students can input their experimental data to calculate various 
    effectiveness metrics.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Screen Cleaner/Grader Theory

        Cleaning and grading are crucial post-harvest operations that remove foreign materials from threshed grains 
        and separate grains into quality fractions based on size, mass, and other characteristics.

        #### Key Concepts:

        **Cleaning:** Removes foreign and undesirable materials from the desired grains using methods such as:
        - Screening
        - Air separation (winnowing)
        - Washing
        - Hand picking

        **Grading:** Classifies cleaned products into various quality fractions based on:
        - Size
        - Mass
        - Shape
        - Color

        #### Working Principle:

        A cleaner cum grader typically works on the principle of vibration and air separation:
        - A set of sieves is oscillated by an eccentric mechanism
        - Grains are graded based on sieve perforation size
        - Simultaneously, a blower forces air onto the grain to remove chaff and lightweight impurities

        The effectiveness of cleaning is determined by analyzing the fraction of good grains and chaff in the:
        - Feed (input material)
        - Good grain outlet
        - Chaff/impurity outlet

        #### Effectiveness Calculations:

        Let's define the key parameters:
        - X: Mass fraction of good grains in the feed
        - Y: Mass fraction of good grains in the good grain outlet
        - Z: Mass fraction of good grains in the chaff/impurities outlet

        Where:
        - a, b, c are quantities of good grains in feed, good grain outlet, and chaff outlet respectively
        - A, B, C are total quantities of material in feed, good grain outlet, and chaff outlet respectively

        **Material Balance:** A = B + C

        **Effectiveness with reference to good grains:**
        <div class='formula'>Eg = BY/AX = Y(X-Z)/X(Y-Z)</div>

        **Effectiveness with reference to chaff and impurities:**
        <div class='formula'>Ec = C(1-Z)/A(1-X) = (Y-X)(1-Z)/(Y-Z)(1-X)</div>

        **Overall effectiveness of cleaning:**
        <div class='formula'>Ecl = Eg × Ec = Y(X-Z)(Y-X)(1-Z)/X(Y-Z)(Y-Z)(1-X)</div>

        For grading, the calculation is similar but based on size fractions rather than good grain vs. chaff.
        """, unsafe_allow_html=True)

        # Add diagram of the cleaner
        st.image("https://via.placeholder.com/800x400.png?text=Cleaner+Cum+Grader+Diagram",
                 caption="Fig. Screen Cleaner/Grader Mechanism")

    # Create tabs for different calculations
    tab1, tab2, tab3 = st.tabs(["Cleaning Effectiveness", "Grading Effectiveness", "Data Visualization"])

    with tab1:
        st.markdown("<h3 class='section-header'>Cleaning Effectiveness Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        This calculator determines the effectiveness of the cleaning operation by analyzing samples 
        from the feed, good grain outlet, and chaff/impurity outlet.

        Enter your experimental data below:
        """)

        with st.form("cleaning_effectiveness_form"):
            st.markdown("### Feed Material Samples")

            # Create a table-like interface for feed samples
            col1, col2, col3 = st.columns(3)
            with col1:
                feed_sample1_total = st.number_input("Feed Sample 1 - Total (g)", min_value=0.0, value=50.0, step=0.1)
                feed_sample1_good = st.number_input("Feed Sample 1 - Good Grains (g)", min_value=0.0, value=40.0,
                                                    step=0.1)

            with col2:
                feed_sample2_total = st.number_input("Feed Sample 2 - Total (g)", min_value=0.0, value=50.0, step=0.1)
                feed_sample2_good = st.number_input("Feed Sample 2 - Good Grains (g)", min_value=0.0, value=38.0,
                                                    step=0.1)

            with col3:
                feed_sample3_total = st.number_input("Feed Sample 3 - Total (g)", min_value=0.0, value=50.0, step=0.1)
                feed_sample3_good = st.number_input("Feed Sample 3 - Good Grains (g)", min_value=0.0, value=42.0,
                                                    step=0.1)

            st.markdown("### Good Grain Outlet Samples")

            col1, col2, col3 = st.columns(3)
            with col1:
                good_sample1_total = st.number_input("Good Outlet Sample 1 - Total (g)", min_value=0.0, value=50.0,
                                                     step=0.1)
                good_sample1_good = st.number_input("Good Outlet Sample 1 - Good Grains (g)", min_value=0.0, value=48.0,
                                                    step=0.1)

            with col2:
                good_sample2_total = st.number_input("Good Outlet Sample 2 - Total (g)", min_value=0.0, value=50.0,
                                                     step=0.1)
                good_sample2_good = st.number_input("Good Outlet Sample 2 - Good Grains (g)", min_value=0.0, value=49.0,
                                                    step=0.1)

            with col3:
                good_sample3_total = st.number_input("Good Outlet Sample 3 - Total (g)", min_value=0.0, value=50.0,
                                                     step=0.1)
                good_sample3_good = st.number_input("Good Outlet Sample 3 - Good Grains (g)", min_value=0.0, value=47.5,
                                                    step=0.1)

            st.markdown("### Chaff/Impurity Outlet Samples")

            col1, col2, col3 = st.columns(3)
            with col1:
                chaff_sample1_total = st.number_input("Chaff Outlet Sample 1 - Total (g)", min_value=0.0, value=50.0,
                                                      step=0.1)
                chaff_sample1_good = st.number_input("Chaff Outlet Sample 1 - Good Grains (g)", min_value=0.0,
                                                     value=5.0, step=0.1)

            with col2:
                chaff_sample2_total = st.number_input("Chaff Outlet Sample 2 - Total (g)", min_value=0.0, value=50.0,
                                                      step=0.1)
                chaff_sample2_good = st.number_input("Chaff Outlet Sample 2 - Good Grains (g)", min_value=0.0,
                                                     value=4.5, step=0.1)

            with col3:
                chaff_sample3_total = st.number_input("Chaff Outlet Sample 3 - Total (g)", min_value=0.0, value=50.0,
                                                      step=0.1)
                chaff_sample3_good = st.number_input("Chaff Outlet Sample 3 - Good Grains (g)", min_value=0.0,
                                                     value=6.0, step=0.1)

            calculate_button = st.form_submit_button("Calculate Cleaning Effectiveness")

        if calculate_button:
            # Calculate average values
            feed_total_avg = (feed_sample1_total + feed_sample2_total + feed_sample3_total) / 3
            feed_good_avg = (feed_sample1_good + feed_sample2_good + feed_sample3_good) / 3

            good_total_avg = (good_sample1_total + good_sample2_total + good_sample3_total) / 3
            good_good_avg = (good_sample1_good + good_sample2_good + good_sample3_good) / 3

            chaff_total_avg = (chaff_sample1_total + chaff_sample2_total + chaff_sample3_total) / 3
            chaff_good_avg = (chaff_sample1_good + chaff_sample2_good + chaff_sample3_good) / 3

            # Calculate mass fractions
            X = feed_good_avg / feed_total_avg  # Mass fraction of good grains in feed
            Y = good_good_avg / good_total_avg  # Mass fraction of good grains in good outlet
            Z = chaff_good_avg / chaff_total_avg  # Mass fraction of good grains in chaff outlet

            # Calculate effectiveness
            Eg = (Y * (X - Z)) / (X * (Y - Z))  # Effectiveness with reference to good grains
            Ec = ((Y - X) * (1 - Z)) / ((Y - Z) * (1 - X))  # Effectiveness with reference to chaff
            Ecl = Eg * Ec  # Overall effectiveness of cleaning

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Mass Fraction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**X (Feed):** {X:.4f}")
            with col2:
                st.markdown(f"**Y (Good Outlet):** {Y:.4f}")
            with col3:
                st.markdown(f"**Z (Chaff Outlet):** {Z:.4f}")

            st.markdown("### Effectiveness Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Eg (Good Grains):** {Eg:.4f}")
            with col2:
                st.markdown(f"**Ec (Chaff):** {Ec:.4f}")
            with col3:
                st.markdown(f"**Ecl (Overall):** {Ecl:.4f}")

            # Convert to percentage for display
            Eg_percent = Eg * 100
            Ec_percent = Ec * 100
            Ecl_percent = Ecl * 100

            st.markdown(f"### Overall Cleaning Effectiveness: {Ecl_percent:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

            # Create visualization
            st.markdown("<h4>Visualization of Effectiveness:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = ['Eg (Good Grains)', 'Ec (Chaff)', 'Ecl (Overall)']
            values = [Eg_percent, Ec_percent, Ecl_percent]
            colors = ['green', 'red', 'blue']

            bars = ax.bar(metrics, values, color=colors)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom')

            ax.set_ylim(0, 110)  # Set y-axis limit to accommodate labels
            ax.set_ylabel('Effectiveness (%)')
            ax.set_title('Cleaning Effectiveness Metrics')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Add interpretation
            st.markdown("""
            ### Interpretation of Results:

            - **Effectiveness with reference to good grains (Eg)**: Measures how effectively good grains are directed to the good grain outlet.
            - **Effectiveness with reference to chaff (Ec)**: Measures how effectively chaff and impurities are directed to the chaff outlet.
            - **Overall effectiveness (Ecl)**: Combined measure of the cleaning operation's success.

            Ideal values approach 1.0 (or 100%), indicating perfect separation.

            Common issues affecting effectiveness:
            - Improper air flow rate
            - Inappropriate sieve perforation size
            - Overloading of the feed hopper
            - Improper machine settings or maintenance
            """)

    with tab2:
        st.markdown("<h3 class='section-header'>Grading Effectiveness Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        This calculator determines the effectiveness of the grading operation by analyzing size-based separation 
        into overflow and underflow fractions.

        Enter your experimental data below:
        """)

        with st.form("grading_effectiveness_form"):
            st.markdown("### Feed Material Samples")

            col1, col2, col3 = st.columns(3)
            with col1:
                grade_feed_sample1_total = st.number_input("Feed Sample 1 - Total (g)", min_value=0.0, value=50.0,
                                                           step=0.1, key="gf1t")
                grade_feed_sample1_desired = st.number_input("Feed Sample 1 - Desired Size (g)", min_value=0.0,
                                                             value=30.0, step=0.1, key="gf1d")

            with col2:
                grade_feed_sample2_total = st.number_input("Feed Sample 2 - Total (g)", min_value=0.0, value=50.0,
                                                           step=0.1, key="gf2t")
                grade_feed_sample2_desired = st.number_input("Feed Sample 2 - Desired Size (g)", min_value=0.0,
                                                             value=28.0, step=0.1, key="gf2d")

            with col3:
                grade_feed_sample3_total = st.number_input("Feed Sample 3 - Total (g)", min_value=0.0, value=50.0,
                                                           step=0.1, key="gf3t")
                grade_feed_sample3_desired = st.number_input("Feed Sample 3 - Desired Size (g)", min_value=0.0,
                                                             value=32.0, step=0.1, key="gf3d")

            st.markdown("### Overflow Samples (Size above sieve opening)")

            col1, col2, col3 = st.columns(3)
            with col1:
                over_sample1_total = st.number_input("Overflow Sample 1 - Total (g)", min_value=0.0, value=25.0,
                                                     step=0.1, key="of1t")
                over_sample1_desired = st.number_input("Overflow Sample 1 - Desired Size (g)", min_value=0.0,
                                                       value=23.0, step=0.1, key="of1d")

            with col2:
                over_sample2_total = st.number_input("Overflow Sample 2 - Total (g)", min_value=0.0, value=26.0,
                                                     step=0.1, key="of2t")
                over_sample2_desired = st.number_input("Overflow Sample 2 - Desired Size (g)", min_value=0.0,
                                                       value=24.0, step=0.1, key="of2d")

            with col3:
                over_sample3_total = st.number_input("Overflow Sample 3 - Total (g)", min_value=0.0, value=24.0,
                                                     step=0.1, key="of3t")
                over_sample3_desired = st.number_input("Overflow Sample 3 - Desired Size (g)", min_value=0.0,
                                                       value=22.0, step=0.1, key="of3d")

            st.markdown("### Underflow Samples (Size below sieve opening)")

            col1, col2, col3 = st.columns(3)
            with col1:
                under_sample1_total = st.number_input("Underflow Sample 1 - Total (g)", min_value=0.0, value=25.0,
                                                      step=0.1, key="uf1t")
                under_sample1_desired = st.number_input("Underflow Sample 1 - Desired Size (g)", min_value=0.0,
                                                        value=7.0, step=0.1, key="uf1d")

            with col2:
                under_sample2_total = st.number_input("Underflow Sample 2 - Total (g)", min_value=0.0, value=24.0,
                                                      step=0.1, key="uf2t")
                under_sample2_desired = st.number_input("Underflow Sample 2 - Desired Size (g)", min_value=0.0,
                                                        value=6.0, step=0.1, key="uf2d")

            with col3:
                under_sample3_total = st.number_input("Underflow Sample 3 - Total (g)", min_value=0.0, value=26.0,
                                                      step=0.1, key="uf3t")
                under_sample3_desired = st.number_input("Underflow Sample 3 - Desired Size (g)", min_value=0.0,
                                                        value=8.0, step=0.1, key="uf3d")

            # Input for sieve opening size
            sieve_size = st.number_input("Sieve Opening Size (mm)", min_value=0.1, value=2.0, step=0.1)

            calculate_grading_button = st.form_submit_button("Calculate Grading Effectiveness")

        if calculate_grading_button:
            # Calculate average values
            grade_feed_total_avg = (grade_feed_sample1_total + grade_feed_sample2_total + grade_feed_sample3_total) / 3
            grade_feed_desired_avg = (
                                                 grade_feed_sample1_desired + grade_feed_sample2_desired + grade_feed_sample3_desired) / 3

            over_total_avg = (over_sample1_total + over_sample2_total + over_sample3_total) / 3
            over_desired_avg = (over_sample1_desired + over_sample2_desired + over_sample3_desired) / 3

            under_total_avg = (under_sample1_total + under_sample2_total + under_sample3_total) / 3
            under_desired_avg = (under_sample1_desired + under_sample2_desired + under_sample3_desired) / 3

            # Calculate mass fractions for grading
            X_grade = grade_feed_desired_avg / grade_feed_total_avg  # Mass fraction of desired size in feed
            Y_grade = over_desired_avg / over_total_avg  # Mass fraction of desired size in overflow
            Z_grade = under_desired_avg / under_total_avg  # Mass fraction of desired size in underflow

            # Calculate effectiveness metrics for grading
            Eo = (Y_grade * (X_grade - Z_grade)) / (
                        X_grade * (Y_grade - Z_grade))  # Effectiveness with reference to overflow
            Eu = ((Y_grade - X_grade) * (1 - Z_grade)) / (
                        (Y_grade - Z_grade) * (1 - X_grade))  # Effectiveness with reference to underflow
            Eg = Eo * Eu  # Overall effectiveness of grading

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Mass Fraction Results for Grading")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**X (Feed):** {X_grade:.4f}")
            with col2:
                st.markdown(f"**Y (Overflow):** {Y_grade:.4f}")
            with col3:
                st.markdown(f"**Z (Underflow):** {Z_grade:.4f}")

            st.markdown("### Effectiveness Results for Grading")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Eo (Overflow):** {Eo:.4f}")
            with col2:
                st.markdown(f"**Eu (Underflow):** {Eu:.4f}")
            with col3:
                st.markdown(f"**Eg (Overall):** {Eg:.4f}")

            # Convert to percentage for display
            Eo_percent = Eo * 100
            Eu_percent = Eu * 100
            Eg_percent = Eg * 100

            st.markdown(f"### Overall Grading Effectiveness: {Eg_percent:.2f}%")
            st.markdown(f"**Sieve Opening Size:** {sieve_size} mm")

            st.markdown("</div>", unsafe_allow_html=True)

            # Create visualization
            st.markdown("<h4>Visualization of Grading Effectiveness:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = ['Eo (Overflow)', 'Eu (Underflow)', 'Eg (Overall)']
            values = [Eo_percent, Eu_percent, Eg_percent]
            colors = ['orange', 'purple', 'teal']

            bars = ax.bar(metrics, values, color=colors)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom')

            ax.set_ylim(0, 110)  # Set y-axis limit to accommodate labels
            ax.set_ylabel('Effectiveness (%)')
            ax.set_title('Grading Effectiveness Metrics')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Add interpretation
            st.markdown("""
            ### Interpretation of Grading Results:

            - **Effectiveness with reference to overflow (Eo)**: Measures how effectively the desired size particles are directed to the overflow.
            - **Effectiveness with reference to underflow (Eu)**: Measures how effectively smaller particles are directed to the underflow.
            - **Overall grading effectiveness (Eg)**: Combined measure of the grading operation's success.

            Factors affecting grading effectiveness:
            - Sieve opening size
            - Oscillation frequency and amplitude
            - Feeding rate
            - Particle shape and moisture content
            """)

            # Show size distribution visualization
            st.markdown("<h4>Size Distribution Analysis:</h4>", unsafe_allow_html=True)

            # Create data for size distribution chart
            categories = ['Feed', 'Overflow', 'Underflow']
            desired_size = [X_grade * 100, Y_grade * 100, Z_grade * 100]
            other_size = [100 - x for x in desired_size]

            fig, ax = plt.subplots(figsize=(10, 6))

            width = 0.35
            x = np.arange(len(categories))

            ax.bar(x, desired_size, width, label='Desired Size', color='green')
            ax.bar(x, other_size, width, bottom=desired_size, label='Other Size', color='gray')

            ax.set_ylabel('Percentage (%)')
            ax.set_title('Size Distribution in Different Fractions')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        st.markdown("<h3 class='section-header'>Data Visualization and Analysis</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section provides advanced visualizations to help understand the performance of the screen cleaner/grader.
        You can visualize material flow, perform sensitivity analysis, and explore the relationship between different parameters.
        """)

        # Create a sample data generator for demonstration
        st.markdown("### Sample Data Generator")
        st.markdown(
            "Adjust the parameters below to generate sample data and see how they affect cleaning and grading effectiveness.")

        col1, col2 = st.columns(2)

        with col1:
            sample_X = st.slider("X: Mass Fraction in Feed", 0.0, 1.0, 0.8, 0.01)
            sample_Y = st.slider("Y: Mass Fraction in Good Outlet", sample_X, 1.0, 0.95, 0.01)
            sample_Z = st.slider("Z: Mass Fraction in Chaff Outlet", 0.0, sample_X, 0.1, 0.01)

        with col2:
            air_flow = st.slider("Air Flow Rate (m³/min)", 1.0, 10.0, 5.0, 0.1)
            feed_rate = st.slider("Feed Rate (kg/h)", 10, 200, 60, 5)
            sieve_oscillation = st.slider("Sieve Oscillation (Hz)", 1.0, 10.0, 5.0, 0.1)

        # Calculate effectiveness based on sample data
        sample_Eg = (sample_Y * (sample_X - sample_Z)) / (sample_X * (sample_Y - sample_Z))
        sample_Ec = ((sample_Y - sample_X) * (1 - sample_Z)) / ((sample_Y - sample_Z) * (1 - sample_X))
        sample_Ecl = sample_Eg * sample_Ec

        # Display calculated values
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        st.markdown("### Calculated Effectiveness Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Eg (Good Grains):** {sample_Eg:.4f}")
        with col2:
            st.markdown(f"**Ec (Chaff):** {sample_Ec:.4f}")
        with col3:
            st.markdown(f"**Ecl (Overall):** {sample_Ecl:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Create Sankey diagram for material flow
        st.markdown("### Material Flow Visualization")

        # Create placeholder for Sankey diagram (would be implemented with plotly in a full application)
        st.image("https://via.placeholder.com/800x400.png?text=Sankey+Diagram+of+Material+Flow",
                 caption="Fig. Material Flow Through Cleaner/Grader")

        # Create sensitivity analysis visualization
        st.markdown("### Sensitivity Analysis")
        st.markdown("This chart shows how changes in input parameters affect the overall effectiveness.")

        # Generate data for sensitivity analysis
        x_values = np.linspace(0.6, 0.9, 20)
        y_values = np.linspace(0.85, 0.99, 20)
        z_values = np.linspace(0.05, 0.2, 20)

        effectiveness_data = []

        for x in x_values:
            Eg_list = []
            for y in [0.95]:  # Fixed Y for this example
                for z in [0.1]:  # Fixed Z for this example
                    if x < y and z < x:  # Valid condition
                        Eg = (y * (x - z)) / (x * (y - z))
                        Eg_list.append(Eg)
                    else:
                        Eg_list.append(None)
            effectiveness_data.append(Eg_list[0] if Eg_list else None)

        # Create the chart
        fig, ax = plt.subplots(figsize=(10, 6))

        valid_indices = [i for i, v in enumerate(effectiveness_data) if v is not None]
        valid_x = [x_values[i] for i in valid_indices]
        valid_e = [effectiveness_data[i] for i in valid_indices]

        ax.plot(valid_x, valid_e, 'b-', linewidth=2, label='Eg (Y=0.95, Z=0.1)')
        ax.set_xlabel('X: Mass Fraction in Feed')
        ax.set_ylabel('Effectiveness (Eg)')
        ax.set_title('Sensitivity of Effectiveness to Feed Quality')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Create 3D visualization
        st.markdown("### 3D Relationship Visualization")
        st.markdown(
            "This interactive visualization shows the relationship between feed rate, air flow, and effectiveness.")

        # Create placeholder for 3D visualization (would be implemented with plotly in a full application)
        st.image("https://via.placeholder.com/800x400.png?text=3D+Visualization+of+Parameter+Relationships",
                 caption="Fig. 3D Relationship between Operating Parameters and Effectiveness")

        # Add capacity calculator
        st.markdown("### Capacity Calculator")
        st.markdown("Calculate the capacity of the cleaner cum grader.")

        with st.form("capacity_form"):
            col1, col2 = st.columns(2)

            with col1:
                sample_mass = st.number_input("Sample Mass (kg)", min_value=0.1, value=1.0, step=0.1)
                processing_time = st.number_input("Processing Time (min)", min_value=0.1, value=1.0, step=0.1)

            with col2:
                num_runs = st.number_input("Number of Runs", min_value=1, value=3, step=1)
                st.markdown(
                    "**Note:** For accurate capacity measurement, use a larger sample and measure the time precisely.")

            calculate_capacity_button = st.form_submit_button("Calculate Capacity")

        if calculate_capacity_button:
            # Calculate capacity
            capacity = (sample_mass / processing_time) * 60  # kg/h

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"### Calculated Capacity: {capacity:.2f} kg/h")
            st.markdown("</div>", unsafe_allow_html=True)

# Tray Dryer Evaluation Module
elif page == "Tray Dryer Evaluation":
    st.markdown("<h2 class='sub-header'>Performance Evaluation of Tray Dryer</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This module helps evaluate the performance of tray dryers by drawing drying characteristic curves 
    and calculating key performance metrics such as heat utilization factor, coefficient of performance, 
    and drying constant.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Tray Dryer Theory

        Tray drying is a process of drying materials placed in trays within a heated chamber. Hot air circulates 
        through the chamber, removing moisture through both conduction and convection mechanisms.

        #### Key Performance Parameters:

        **1. Heat Utilization Factor (HUF):**
        <div class='formula'>HUF = Heat utilized / Heat supplied = (t₁-t₂)/(t₁-t₀)</div>

        Where:
        - t₀ = Dry bulb temperature of ambient air (°C)
        - t₁ = Dry bulb temperature of drying air (°C)
        - t₂ = Dry bulb temperature of exhaust air (°C)

        **2. Coefficient of Performance (COP):**
        <div class='formula'>COP = (t₂-t₀)/(t₁-t₀)</div>

        **3. Drying Kinetics (Newton's Law of Cooling):**
        <div class='formula'>(M-Me)/(M₀-Me) = e^(-kθ)</div>

        Rearranged to find the drying constant:
        <div class='formula'>K = (1/θ) × ln((M₀-Me)/(M-Me))</div>

        Where:
        - M = Moisture content at time θ (% db)
        - M₀ = Initial moisture content (% db)
        - Me = Equilibrium moisture content (% db)
        - θ = Drying time (h)
        - K = Drying constant (1/h)

        #### Operating Parameters:

        - **Air Temperature:** Typically 50-93°C for food materials
        - **Air Velocity:** Usually 2.5-5 m/s across trays
        - **Tray Configuration:** Multiple shallow trays stacked with gaps for air flow
        - **Layer Thickness:** Thin layers (few centimeters) for uniform drying

        #### Advantages of Tray Dryers:

        - Flexibility to accommodate various food products
        - Suitable for small-scale production
        - Low capital and maintenance costs

        #### Disadvantages:

        - Batch system operation
        - Non-uniform drying of products
        - Variable product quality as food dries more rapidly on trays nearest to the heat source
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.image("https://via.placeholder.com/400x300.png?text=Tray+Dryer+External+View",
                     caption="External view of a typical tray dryer")

        with col2:
            st.image("https://via.placeholder.com/400x300.png?text=Tray+Dryer+Schematic",
                     caption="Schematic diagram of tray dryer showing air flow")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Moisture Content Analysis", "Performance Metrics", "Drying Curves"])

    with tab1:
        st.markdown("<h3 class='section-header'>Moisture Content Determination</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section helps calculate the moisture content of the grain/seed samples before and during 
        the drying process.
        """)

        with st.form("moisture_content_form"):
            st.markdown("### Sample Information")

            grain_type = st.text_input("Grain/Seed Type", "Wheat")

            col1, col2 = st.columns(2)
            with col1:
                cup_empty_weight = st.number_input("Empty Cup Weight (g)", min_value=0.0, value=10.0, step=0.1)
                cup_with_wet_sample = st.number_input("Cup + Wet Sample Weight (g)", min_value=0.0, value=15.0,
                                                      step=0.1)

            with col2:
                cup_with_dry_sample = st.number_input("Cup + Dry Sample Weight (g)", min_value=0.0, value=13.5,
                                                      step=0.1)
                sample_area = st.number_input("Sample Tray Area (cm²)", min_value=1.0, value=100.0, step=1.0)

            initial_sample_weight = st.number_input("Initial Sample Weight for Drying (g)", min_value=0.0, value=100.0,
                                                    step=0.1)

            calculate_mc_button = st.form_submit_button("Calculate Initial Moisture Content")

        if calculate_mc_button:
            # Calculate sample weights
            wet_sample_weight = cup_with_wet_sample - cup_empty_weight
            dry_sample_weight = cup_with_dry_sample - cup_empty_weight
            moisture_weight = wet_sample_weight - dry_sample_weight

            # Calculate moisture content (wet basis)
            moisture_content_wb = (moisture_weight / wet_sample_weight) * 100

            # Calculate moisture content (dry basis)
            moisture_content_db = (moisture_weight / dry_sample_weight) * 100

            # Calculate probable dry weight of the initial sample
            probable_dry_weight = (initial_sample_weight * 100) / (100 + moisture_content_db)

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Moisture Content Results")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Wet Sample Weight:** {wet_sample_weight:.2f} g")
                st.markdown(f"**Dry Sample Weight:** {dry_sample_weight:.2f} g")
                st.markdown(f"**Moisture Weight:** {moisture_weight:.2f} g")

            with col2:
                st.markdown(f"**Moisture Content (wet basis):** {moisture_content_wb:.2f}%")
                st.markdown(f"**Moisture Content (dry basis):** {moisture_content_db:.2f}%")
                st.markdown(f"**Probable Dry Weight of Initial Sample:** {probable_dry_weight:.2f} g")

            st.markdown("</div>", unsafe_allow_html=True)

            # Store values in session state for use in other tabs
            st.session_state.initial_moisture_db = moisture_content_db
            st.session_state.initial_sample_weight = initial_sample_weight
            st.session_state.probable_dry_weight = probable_dry_weight
            st.session_state.sample_area = sample_area

    with tab2:
        st.markdown("<h3 class='section-header'>Performance Metrics Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section calculates the Heat Utilization Factor (HUF), Coefficient of Performance (COP), 
        and other performance metrics based on temperature readings.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Temperature Measurements")

            ambient_temp = st.number_input("Ambient Air Temperature (t₀) °C", min_value=0.0, max_value=50.0, value=25.0,
                                           step=0.1)
            drying_temp = st.number_input("Drying Air Temperature (t₁) °C", min_value=ambient_temp, max_value=150.0,
                                          value=60.0, step=0.1)
            exhaust_temp = st.number_input("Exhaust Air Temperature (t₂) °C", min_value=ambient_temp,
                                           max_value=drying_temp, value=45.0, step=0.1)

        with col2:
            st.markdown("### Additional Parameters")

            power_input = st.number_input("Heater Power Input (W)", min_value=0, value=1000, step=100)
            air_flow_rate = st.number_input("Air Flow Rate (m³/min)", min_value=0.0, value=1.5, step=0.1)
            drying_time = st.number_input("Total Drying Time (min)", min_value=0, value=120, step=5)

        if st.button("Calculate Performance Metrics"):
            # Calculate HUF
            huf = (drying_temp - exhaust_temp) / (drying_temp - ambient_temp)

            # Calculate COP
            cop = (exhaust_temp - ambient_temp) / (drying_temp - ambient_temp)

            # Calculate energy used
            energy_used_kwh = (power_input * drying_time) / (60 * 1000)  # Convert to kWh

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Performance Metrics Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Heat Utilization Factor (HUF):** {huf:.4f}")
            with col2:
                st.markdown(f"**Coefficient of Performance (COP):** {cop:.4f}")
            with col3:
                st.markdown(f"**Energy Consumption:** {energy_used_kwh:.3f} kWh")

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = ['HUF', 'COP']
            values = [huf, cop]
            colors = ['blue', 'green']

            bars = ax.bar(metrics, values, color=colors)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

            ax.set_ylim(0, max(values) * 1.2)  # Set y-axis limit with some headroom
            ax.set_ylabel('Value')
            ax.set_title('Dryer Performance Metrics')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Temperature comparison visualization
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            temp_points = ['Ambient (t₀)', 'Drying (t₁)', 'Exhaust (t₂)']
            temp_values = [ambient_temp, drying_temp, exhaust_temp]
            temp_colors = ['lightblue', 'red', 'orange']

            bars2 = ax2.bar(temp_points, temp_values, color=temp_colors)

            # Add value labels on top of bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{height:.1f}°C', ha='center', va='bottom')

            ax2.set_ylabel('Temperature (°C)')
            ax2.set_title('Temperature Profile')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("</div>", unsafe_allow_html=True)

            # Interpretation of results
            st.markdown("""
            ### Interpretation of Results:

            - **Heat Utilization Factor (HUF):** Indicates how efficiently the heat is being utilized in the dryer. Higher values (closer to 1.0) indicate better heat utilization.
            - **Coefficient of Performance (COP):** Measures the efficiency of the drying process. Higher values indicate more efficient operation.

            Factors affecting performance:
            - Air flow rate and distribution
            - Insulation quality of the drying chamber
            - Product loading density on trays
            - Initial moisture content of the product
            - Ambient conditions
            """)

    with tab3:
        st.markdown("<h3 class='section-header'>Drying Characteristic Curves</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section helps create and analyze drying characteristic curves based on moisture content measurements 
        over time. You can either input your experimental data or use a simulation for demonstration.
        """)

        input_method = st.radio(
            "Select Input Method",
            ["Use Experimental Data", "Run Simulation"]
        )

        if input_method == "Use Experimental Data":
            st.markdown("### Enter Your Experimental Data")

            # Create a form for entering time-series data
            with st.form("drying_data_form"):
                num_readings = st.number_input("Number of Time Points", min_value=2, max_value=20, value=6, step=1)

                # Create columns for time and mass inputs
                col1, col2, col3, col4 = st.columns(4)

                times = []
                masses = []
                ambient_temps = []
                drying_temps = []
                exhaust_temps = []

                with col1:
                    st.markdown("**Time (min)**")
                    for i in range(num_readings):
                        times.append(
                            st.number_input(f"Time {i + 1}", min_value=0, value=i * 15, step=5, key=f"time_{i}"))

                with col2:
                    st.markdown("**Sample Mass (g)**")
                    for i in range(num_readings):
                        if i == 0:
                            default_mass = 100.0
                        else:
                            default_mass = max(100.0 - i * 5, 80.0)  # A simple decreasing pattern
                        masses.append(st.number_input(f"Mass {i + 1}", min_value=0.0, value=default_mass, step=0.1,
                                                      key=f"mass_{i}"))

                with col3:
                    st.markdown("**Ambient Temp (°C)**")
                    for i in range(num_readings):
                        ambient_temps.append(
                            st.number_input(f"t₀ {i + 1}", min_value=0.0, value=25.0, step=0.1, key=f"t0_{i}"))

                    st.markdown("**Equil. Moisture (%db)**")
                    equilibrium_moisture = st.number_input("Me", min_value=0.0, max_value=20.0, value=8.0, step=0.1)

                with col4:
                    st.markdown("**Drying Air Temp (°C)**")
                    for i in range(num_readings):
                        drying_temps.append(
                            st.number_input(f"t₁ {i + 1}", min_value=0.0, value=60.0, step=0.1, key=f"t1_{i}"))

                    st.markdown("**Exhaust Air Temp (°C)**")
                    for i in range(num_readings):
                        exhaust_temps.append(
                            st.number_input(f"t₂ {i + 1}", min_value=0.0, value=45.0, step=0.1, key=f"t2_{i}"))

                plot_button = st.form_submit_button("Generate Drying Curves")

            if plot_button:
                # Calculate dry basis moisture content if we have initial moisture data
                if 'probable_dry_weight' in st.session_state and 'initial_sample_weight' in st.session_state:
                    dry_weight = st.session_state.probable_dry_weight
                    moisture_contents = [(m - dry_weight) / dry_weight * 100 for m in masses]
                else:
                    # Estimate if we don't have initial data
                    dry_weight = masses[-1] * 100 / (100 + equilibrium_moisture)
                    moisture_contents = [(m - dry_weight) / dry_weight * 100 for m in masses]

                # Calculate HUF and COP for each time point
                hufs = [(t1 - t2) / (t1 - t0) for t0, t1, t2 in zip(ambient_temps, drying_temps, exhaust_temps)]
                cops = [(t2 - t0) / (t1 - t0) for t0, t1, t2 in zip(ambient_temps, drying_temps, exhaust_temps)]

                # Calculate drying constants between consecutive time points
                drying_constants = []
                for i in range(1, len(times)):
                    if moisture_contents[i] <= equilibrium_moisture or moisture_contents[i - 1] <= equilibrium_moisture:
                        k = 0  # Avoid division by zero or negative values
                    else:
                        time_diff_hours = (times[i] - times[i - 1]) / 60  # Convert minutes to hours
                        k = (1 / time_diff_hours) * np.log((moisture_contents[i - 1] - equilibrium_moisture) /
                                                           (moisture_contents[i] - equilibrium_moisture))
                    drying_constants.append(k)

                # Calculate average drying constant
                avg_drying_constant = np.mean([k for k in drying_constants if k > 0]) if any(
                    k > 0 for k in drying_constants) else 0

                # Create dataframe for displaying results
                data = {
                    'Time (min)': times,
                    'Sample Mass (g)': masses,
                    'Moisture Content (%db)': moisture_contents,
                    'Ambient Temp (°C)': ambient_temps,
                    'Drying Temp (°C)': drying_temps,
                    'Exhaust Temp (°C)': exhaust_temps,
                    'HUF': hufs,
                    'COP': cops
                }

                if len(drying_constants) > 0:
                    data['Drying Constant K (1/h)'] = [0] + drying_constants

                results_df = pd.DataFrame(data)

                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("### Drying Process Data")
                st.dataframe(results_df.style.format({
                    'Moisture Content (%db)': '{:.2f}',
                    'HUF': '{:.3f}',
                    'COP': '{:.3f}',
                    'Drying Constant K (1/h)': '{:.4f}'
                }))

                st.markdown(f"**Average Drying Constant (K):** {avg_drying_constant:.4f} 1/h")
                st.markdown("</div>", unsafe_allow_html=True)

                # Create drying characteristic curve
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(times, moisture_contents, 'o-', color='blue', linewidth=2)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Moisture Content (% db)')
                ax.set_title('Drying Characteristic Curve')
                ax.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

                # Create temperature profile plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                ax2.plot(times, ambient_temps, 'o-', label='Ambient (t₀)', color='green')
                ax2.plot(times, drying_temps, 'o-', label='Drying (t₁)', color='red')
                ax2.plot(times, exhaust_temps, 'o-', label='Exhaust (t₂)', color='orange')

                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('Temperature (°C)')
                ax2.set_title('Temperature Profile During Drying')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig2)

                # Create HUF and COP plot
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                ax3.plot(times, hufs, 'o-', label='HUF', color='blue')
                ax3.plot(times, cops, 'o-', label='COP', color='green')

                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Value')
                ax3.set_title('HUF and COP During Drying')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()

                plt.tight_layout()
                st.pyplot(fig3)

                # Create drying rate curve (negative derivative of moisture content)
                if len(times) > 1:
                    drying_rates = []
                    midpoint_times = []

                    for i in range(1, len(times)):
                        time_diff = (times[i] - times[i - 1]) / 60  # hours
                        mc_diff = moisture_contents[i - 1] - moisture_contents[i]  # % db
                        drying_rate = mc_diff / time_diff if time_diff > 0 else 0  # % db/h
                        drying_rates.append(drying_rate)
                        midpoint_times.append((times[i] + times[i - 1]) / 2)

                    fig4, ax4 = plt.subplots(figsize=(10, 6))

                    ax4.plot(midpoint_times, drying_rates, 'o-', color='purple', linewidth=2)
                    ax4.set_xlabel('Time (min)')
                    ax4.set_ylabel('Drying Rate (% db/h)')
                    ax4.set_title('Drying Rate Curve')
                    ax4.grid(True, linestyle='--', alpha=0.7)

                    plt.tight_layout()
                    st.pyplot(fig4)

        else:  # Simulation mode
            st.markdown("### Drying Process Simulation")
            st.markdown("""
            Adjust the parameters below to simulate a drying process and generate characteristic curves.
            This simulation uses Newton's Law of Cooling for drying kinetics.
            """)

            col1, col2 = st.columns(2)

            with col1:
                initial_mc = st.slider("Initial Moisture Content (% db)", 10.0, 100.0, 50.0, 0.5)
                equilibrium_mc = st.slider("Equilibrium Moisture Content (% db)", 1.0, 20.0, 8.0, 0.5)
                drying_constant = st.slider("Drying Constant K (1/h)", 0.05, 2.0, 0.5, 0.01)

            with col2:
                drying_temp_sim = st.slider("Drying Air Temperature (°C)", 40.0, 100.0, 60.0, 1.0)
                ambient_temp_sim = st.slider("Ambient Temperature (°C)", 10.0, 35.0, 25.0, 1.0)
                simulation_time = st.slider("Simulation Time (h)", 0.5, 10.0, 4.0, 0.5)

            if st.button("Run Simulation"):
                # Generate time points
                time_hours = np.linspace(0, simulation_time, 50)
                time_mins = time_hours * 60

                # Calculate moisture content using Newton's Law of Cooling
                moisture_contents = []
                for t in time_hours:
                    mc = equilibrium_mc + (initial_mc - equilibrium_mc) * np.exp(-drying_constant * t)
                    moisture_contents.append(mc)

                # Simulate exhaust temperature (simplified model)
                exhaust_temps_sim = []
                for mc in moisture_contents:
                    # Exhaust temp decreases as moisture is removed (simplified relationship)
                    moisture_factor = (mc - equilibrium_mc) / (
                                initial_mc - equilibrium_mc) if initial_mc != equilibrium_mc else 0
                    temp_drop = (drying_temp_sim - ambient_temp_sim) * (0.3 + 0.4 * moisture_factor)
                    exhaust_temp = drying_temp_sim - temp_drop
                    exhaust_temps_sim.append(exhaust_temp)

                # Calculate HUF and COP
                hufs_sim = [(drying_temp_sim - et) / (drying_temp_sim - ambient_temp_sim) for et in exhaust_temps_sim]
                cops_sim = [(et - ambient_temp_sim) / (drying_temp_sim - ambient_temp_sim) for et in exhaust_temps_sim]

                # Create drying characteristic curve
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(time_mins, moisture_contents, '-', color='blue', linewidth=2)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Moisture Content (% db)')
                ax.set_title('Simulated Drying Characteristic Curve')
                ax.grid(True, linestyle='--', alpha=0.7)

                # Add annotation with the drying equation
                equation = f"M = {equilibrium_mc:.1f} + ({initial_mc:.1f} - {equilibrium_mc:.1f})e^(-{drying_constant:.3f}t)"
                ax.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                            ha='center')

                plt.tight_layout()
                st.pyplot(fig)

                # Create temperature profile
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                # Constant temperatures
                ax2.plot(time_mins, [ambient_temp_sim] * len(time_mins), '-', label='Ambient (t₀)', color='green')
                ax2.plot(time_mins, [drying_temp_sim] * len(time_mins), '-', label='Drying (t₁)', color='red')
                # Varying exhaust temperature
                ax2.plot(time_mins, exhaust_temps_sim, '-', label='Exhaust (t₂)', color='orange')

                ax2.set_xlabel('Time (min)')
                ax2.set_ylabel('Temperature (°C)')
                ax2.set_title('Simulated Temperature Profile During Drying')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig2)

                # Create HUF and COP plot
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                ax3.plot(time_mins, hufs_sim, '-', label='HUF', color='blue')
                ax3.plot(time_mins, cops_sim, '-', label='COP', color='green')

                ax3.set_xlabel('Time (min)')
                ax3.set_ylabel('Value')
                ax3.set_title('Simulated HUF and COP During Drying')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()

                plt.tight_layout()
                st.pyplot(fig3)

                # Create drying rate curve
                drying_rates = []
                for t in time_hours:
                    # Derivative of the moisture content equation
                    rate = -drying_constant * (initial_mc - equilibrium_mc) * np.exp(-drying_constant * t)
                    drying_rates.append(-rate)  # Negative sign removed to show positive drying rate

                fig4, ax4 = plt.subplots(figsize=(10, 6))

                ax4.plot(time_mins, drying_rates, '-', color='purple', linewidth=2)
                ax4.set_xlabel('Time (min)')
                ax4.set_ylabel('Drying Rate (% db/h)')
                ax4.set_title('Simulated Drying Rate Curve')
                ax4.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig4)

                # Semi-logarithmic plot to verify exponential behavior
                fig5, ax5 = plt.subplots(figsize=(10, 6))

                # Calculate (M - Me)/(Mo - Me)
                normalized_mc = [(m - equilibrium_mc) / (initial_mc - equilibrium_mc) for m in moisture_contents]

                ax5.semilogy(time_hours, normalized_mc, '-', color='red', linewidth=2)
                ax5.set_xlabel('Time (h)')
                ax5.set_ylabel('(M - Me)/(Mo - Me)')
                ax5.set_title('Semi-logarithmic Plot of Moisture Ratio')
                ax5.grid(True, linestyle='--', alpha=0.7)

                # Add line showing the expected slope
                log_line_x = np.array([0, simulation_time])
                log_line_y = np.exp(-drying_constant * log_line_x)
                ax5.semilogy(log_line_x, log_line_y, '--', color='black', alpha=0.7,
                             label=f'Slope = -K = -{drying_constant:.3f}')
                ax5.legend()

                plt.tight_layout()
                st.pyplot(fig5)

                # Show summary of simulation parameters
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("### Simulation Parameters Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Initial Moisture Content:** {initial_mc:.1f}% db")
                    st.markdown(f"**Equilibrium Moisture Content:** {equilibrium_mc:.1f}% db")

                with col2:
                    st.markdown(f"**Drying Constant (K):** {drying_constant:.3f} 1/h")
                    st.markdown(f"**Drying Time (95% reduction):** {-np.log(0.05) / drying_constant:.2f} h")

                with col3:
                    st.markdown(f"**Drying Air Temperature:** {drying_temp_sim:.1f}°C")
                    st.markdown(f"**Average HUF:** {np.mean(hufs_sim):.3f}")

                st.markdown("</div>", unsafe_allow_html=True)


# Belt Conveyor Evaluation Module
elif page == "Belt Conveyor Evaluation":
    st.markdown(
        "<h2 class='sub-header'>Determination of Capacity of a Belt Conveyor and its Performance Evaluation</h2>",
        unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This module helps determine the capacity and conveying efficiency of belt conveyors used in grain/seed processing.
    You can input belt conveyor parameters and experimental data to calculate theoretical capacity, actual capacity, 
    and conveying efficiency.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Belt Conveyor Theory

        Belt conveyors are important material handling equipment used in grain/seed processing industries.
        They consist of an endless belt operating between two pulleys with its load supported on idlers.

        #### Components of a Belt Conveyor:

        - **Belt**: Usually made of rubberized material with carcass for strength
        - **Drive Mechanism**: Motor and drive pulley to move the belt
        - **End Pulleys/Idlers**: Support the belt and maintain tension
        - **Loading and Discharge Devices**: For adding and removing materials
        - **Supporting Structure**: Frame to hold all components

        #### Advantages of Belt Conveyors:

        - High mechanical efficiency due to antifriction bearings
        - No relative motion between product and belt (minimizes damage)
        - Can operate at higher speeds, allowing large carrying capacity
        - For horizontal transport, belt conveyors can cover longer distances
        - Long service life and low operation costs when properly maintained

        #### Key Parameters for Design:

        - **Belt Width**: Depends on capacity requirement and material characteristics
        - **Trough Angle**: Typically 20° for paddy and most grains (other common angles: 30° and 45°)
        - **Surcharge Angle**: Typically 20° for paddy (range: 5° to 30° for various materials)
        - **Belt Speed**: Determined by material properties and required capacity

        #### Belt Conveyor Capacity Calculations:

        The capacity of a belt conveyor depends on:
        1. Cross-sectional area of material on the belt
        2. Speed of the belt
        3. Bulk density of the material

        **For Troughed Belt Conveyors:**
        - Volume of material per meter length: V = [(a+b)/2] h cm³/m
        - Speed of belt: S = (π D N) / 100 m/min
        - Theoretical capacity: Q theo = (ρ S V 60) 10^-6 kg/h
        - Actual capacity: Q actual = (M / T) 60 kg/h
        - Conveying efficiency: η Conveying = (Q actual / Q theo) × 100%

        Where:
        - a = Bottom width of the trough (cm)
        - b = Top width of the trough (cm)
        - h = Depth of material on belt (cm)
        - D = Diameter of pulley/roller (cm)
        - N = Speed of the roller/pulley (rpm)
        - ρ = Bulk density of the material (kg/m³)
        - M = Mass of material conveyed (kg)
        - T = Time taken to convey the material (min)
        """, unsafe_allow_html=True)

        # Add diagrams
        st.image("https://via.placeholder.com/800x400.png?text=Belt+Conveyor+Components+Diagram",
                 caption="Fig. Components of a belt conveyor system")

        col1, col2 = st.columns(2)
        with col1:
            st.image("https://via.placeholder.com/400x300.png?text=Troughed+Belt+Cross+Section",
                     caption="Cross-section of a troughed belt conveyor")
        with col2:
            st.image("https://via.placeholder.com/400x300.png?text=Belt+Conveyor+Side+View",
                     caption="Side view of a belt conveyor system")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Bulk Density Measurement", "Capacity Calculator", "Performance Analysis"])

    with tab1:
        st.markdown("<h3 class='section-header'>Bulk Density Measurement</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section calculates the bulk density of the material to be conveyed, which is an essential 
        parameter for determining the conveyor capacity.
        """)

        with st.form("bulk_density_form"):
            st.markdown("### Sample Information")

            material_type = st.text_input("Material Type", "Wheat")

            st.markdown("### Bulk Density Measurements")

            col1, col2, col3 = st.columns(3)

            with col1:
                container_volume1 = st.number_input("Container Volume - Sample 1 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0)
                container_mass1 = st.number_input("Container Mass - Sample 1 (g)", min_value=0.0, value=200.0, step=1.0)
                total_mass1 = st.number_input("Container + Sample Mass - Sample 1 (g)", min_value=0.0, value=1000.0,
                                              step=1.0)

            with col2:
                container_volume2 = st.number_input("Container Volume - Sample 2 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0)
                container_mass2 = st.number_input("Container Mass - Sample 2 (g)", min_value=0.0, value=200.0, step=1.0)
                total_mass2 = st.number_input("Container + Sample Mass - Sample 2 (g)", min_value=0.0, value=980.0,
                                              step=1.0)

            with col3:
                container_volume3 = st.number_input("Container Volume - Sample 3 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0)
                container_mass3 = st.number_input("Container Mass - Sample 3 (g)", min_value=0.0, value=200.0, step=1.0)
                total_mass3 = st.number_input("Container + Sample Mass - Sample 3 (g)", min_value=0.0, value=1020.0,
                                              step=1.0)

            calculate_density_button = st.form_submit_button("Calculate Bulk Density")

        if calculate_density_button:
            # Calculate sample masses
            sample_mass1 = total_mass1 - container_mass1
            sample_mass2 = total_mass2 - container_mass2
            sample_mass3 = total_mass3 - container_mass3

            # Calculate bulk densities
            bulk_density1 = sample_mass1 / container_volume1  # g/cc
            bulk_density2 = sample_mass2 / container_volume2  # g/cc
            bulk_density3 = sample_mass3 / container_volume3  # g/cc

            # Calculate average bulk density
            avg_bulk_density_gcc = (bulk_density1 + bulk_density2 + bulk_density3) / 3
            avg_bulk_density_kgm3 = avg_bulk_density_gcc * 1000  # Convert g/cc to kg/m³

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Bulk Density Results")

            # Create a table of results - use None for numeric formatting
            results_data = {
                "Sample": ["Sample 1", "Sample 2", "Sample 3", "Average"],
                "Sample Mass (g)": [sample_mass1, sample_mass2, sample_mass3, None],
                "Volume (cc)": [container_volume1, container_volume2, container_volume3, None],
                "Bulk Density (g/cc)": [bulk_density1, bulk_density2, bulk_density3, avg_bulk_density_gcc],
                "Bulk Density (kg/m³)": [bulk_density1 * 1000, bulk_density2 * 1000, bulk_density3 * 1000,
                                         avg_bulk_density_kgm3]
            }

            results_df = pd.DataFrame(results_data)

            # Display the dataframe without formatting
            st.dataframe(results_df)

            st.markdown(
                f"**The average bulk density of {material_type} is {avg_bulk_density_gcc:.4f} g/cc or {avg_bulk_density_kgm3:.1f} kg/m³**")

            st.markdown("</div>", unsafe_allow_html=True)

            # Store the bulk density in session state for use in other tabs
            st.session_state.bulk_density = avg_bulk_density_kgm3
            st.session_state.material_type = material_type

    with tab2:
        st.markdown("<h3 class='section-header'>Belt Conveyor Capacity Calculator</h3>", unsafe_allow_html=True)

        st.markdown("""
        This calculator determines the theoretical capacity of a belt conveyor based on its dimensions, 
        speed, and the material being conveyed.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Belt & Pulley Parameters")

            pulley_diameter = st.number_input("Diameter of Pulley/Roller (D) [cm]", min_value=1.0, value=30.0, step=0.5)
            pulley_speed = st.number_input("Speed of Roller/Pulley (N) [rpm]", min_value=1.0, value=50.0, step=1.0)
            belt_length = st.number_input("Length of Belt/Distance Conveyed (L) [m]", min_value=0.1, value=3.0,
                                          step=0.1)

        with col2:
            st.markdown("### Trough Parameters")

            bottom_width = st.number_input("Bottom Width of Trough (a) [cm]", min_value=1.0, value=15.0, step=0.5)
            top_width = st.number_input("Top Width of Trough (b) [cm]", min_value=1.0, value=45.0, step=0.5)
            material_depth = st.number_input("Depth of Material on Belt (h) [cm]", min_value=0.1, value=20.0, step=0.5)

        # Get material density from session state or allow manual input
        if 'bulk_density' in st.session_state:
            default_density = st.session_state.bulk_density
            material_name = st.session_state.material_type
        else:
            default_density = 750.0
            material_name = "material"

        material_density = st.number_input(f"Bulk Density of {material_name} (ρ) [kg/m³]",
                                           min_value=100.0, max_value=3000.0, value=default_density, step=10.0)

        if st.button("Calculate Theoretical Capacity"):
            # Calculate volume of material per meter length
            volume_per_meter = ((bottom_width + top_width) / 2) * material_depth  # cm³/m

            # Calculate belt speed
            belt_speed = (np.pi * pulley_diameter * pulley_speed) / 100  # m/min

            # Calculate theoretical capacity
            theoretical_capacity = (material_density * belt_speed * volume_per_meter * 60) * 1e-6  # kg/h

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Theoretical Capacity Results")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Volume per meter length (V):** {volume_per_meter:.2f} cm³/m")
                st.markdown(f"**Belt Speed (S):** {belt_speed:.2f} m/min")

            with col2:
                st.markdown(f"**Theoretical Capacity (Q theo):** {theoretical_capacity:.2f} kg/h")
                st.markdown(f"**Theoretical Capacity:** {theoretical_capacity / 1000:.2f} tonnes/h")

            st.markdown("</div>", unsafe_allow_html=True)

            # Create visualization
            st.markdown("<h4>Belt Conveyor Parameters Visualization:</h4>", unsafe_allow_html=True)

            # Create a cross-section visualization of the belt
            fig, ax = plt.subplots(figsize=(8, 6))

            # Belt cross-section coordinates
            belt_x = [-bottom_width / 2, bottom_width / 2, top_width / 2, -top_width / 2, -bottom_width / 2]
            belt_y = [0, 0, material_depth, material_depth, 0]

            # Plot the belt cross-section
            ax.plot(belt_x, belt_y, 'k-', linewidth=2)

            # Fill the cross-section to represent material
            ax.fill(belt_x, belt_y, color='sandybrown', alpha=0.7)

            # Add labels
            ax.text(0, -5, f"a = {bottom_width} cm", ha='center')
            ax.text(0, material_depth + 5, f"b = {top_width} cm", ha='center')
            ax.text(top_width / 2 + 5, material_depth / 2, f"h = {material_depth} cm", va='center')

            # Add trough angle visualization
            trough_angle = np.arctan((top_width - bottom_width) / (2 * material_depth)) * 180 / np.pi
            ax.text(-top_width / 2 - 10, material_depth / 2, f"Trough Angle: {trough_angle:.1f}°", va='center')

            # Set equal aspect ratio and limits
            ax.set_aspect('equal')
            ax.set_xlim(-top_width / 2 - 20, top_width / 2 + 20)
            ax.set_ylim(-10, material_depth + 20)

            # Remove axes
            ax.axis('off')

            # Add title
            ax.set_title('Belt Conveyor Cross-Section')

            st.pyplot(fig)

            # Create a diagram showing the relationship between key parameters
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Define ranges for speed, volume, and capacity calculation
            speeds = np.linspace(belt_speed * 0.5, belt_speed * 1.5, 100)
            capacities = [(material_density * s * volume_per_meter * 60) * 1e-6 for s in speeds]

            ax2.plot(speeds, capacities, 'b-', linewidth=2)

            # Mark the current values
            ax2.plot(belt_speed, theoretical_capacity, 'ro', markersize=8)
            ax2.annotate(f"({belt_speed:.1f}, {theoretical_capacity:.1f})",
                         (belt_speed, theoretical_capacity),
                         xytext=(10, -15), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", color='red'))

            ax2.set_xlabel('Belt Speed (m/min)')
            ax2.set_ylabel('Theoretical Capacity (kg/h)')
            ax2.set_title('Relationship Between Belt Speed and Capacity')
            ax2.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig2)

            # Store the theoretical capacity in session state for use in the next tab
            st.session_state.theoretical_capacity = theoretical_capacity
            st.session_state.belt_parameters = {
                'pulley_diameter': pulley_diameter,
                'pulley_speed': pulley_speed,
                'belt_length': belt_length,
                'bottom_width': bottom_width,
                'top_width': top_width,
                'material_depth': material_depth,
                'belt_speed': belt_speed,
                'volume_per_meter': volume_per_meter
            }

    with tab3:
        st.markdown("<h3 class='section-header'>Performance Evaluation</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section evaluates the actual performance of the belt conveyor by comparing 
        the theoretical capacity with the actual capacity measured experimentally.
        """)

        # Create a form for the actual capacity measurement
        with st.form("performance_form"):
            st.markdown("### Actual Capacity Measurement")

            col1, col2 = st.columns(2)

            with col1:
                material_mass = st.number_input("Mass of Material Conveyed (M) [kg]", min_value=0.1, value=10.0,
                                                step=0.1)
                conveying_time = st.number_input("Time Taken to Convey the Material (T) [min]", min_value=0.1,
                                                 value=1.0, step=0.1)

            with col2:
                st.markdown("### Reference Parameters")

                # Display theoretical capacity if available
                if 'theoretical_capacity' in st.session_state:
                    st.info(f"Theoretical Capacity: {st.session_state.theoretical_capacity:.2f} kg/h")
                else:
                    st.warning("Calculate theoretical capacity first in the previous tab")

            evaluate_button = st.form_submit_button("Evaluate Performance")

        if evaluate_button:
            # Calculate actual capacity
            actual_capacity = (material_mass / conveying_time) * 60  # kg/h

            # Get theoretical capacity from session state or use a default
            if 'theoretical_capacity' in st.session_state:
                theoretical_capacity = st.session_state.theoretical_capacity
            else:
                theoretical_capacity = 0
                st.warning("Theoretical capacity not found. Please calculate it in the previous tab.")

            # Calculate conveying efficiency
            if theoretical_capacity > 0:
                conveying_efficiency = (actual_capacity / theoretical_capacity) * 100  # percentage
            else:
                conveying_efficiency = 0

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Performance Evaluation Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Theoretical Capacity:** {theoretical_capacity:.2f} kg/h")
            with col2:
                st.markdown(f"**Actual Capacity:** {actual_capacity:.2f} kg/h")
            with col3:
                st.markdown(f"**Conveying Efficiency:** {conveying_efficiency:.2f}%")

            st.markdown("</div>", unsafe_allow_html=True)

            # Explanation based on efficiency
            if conveying_efficiency < 50:
                efficiency_comment = "Low efficiency indicates significant losses or operational issues. Check for material spillage, belt slippage, or loading problems."
            elif conveying_efficiency < 80:
                efficiency_comment = "Moderate efficiency. There may be some operational improvements possible to increase capacity."
            elif conveying_efficiency < 95:
                efficiency_comment = "Good efficiency. The conveyor is performing well with minimal losses."
            elif conveying_efficiency <= 100:
                efficiency_comment = "Excellent efficiency. The conveyor is operating at near-optimal conditions."
            else:
                efficiency_comment = "Efficiency exceeds 100%, which suggests either measurement errors or the theoretical model underestimates capacity."

            st.markdown(f"**Performance Assessment:** {efficiency_comment}")

            # Create visualization
            st.markdown("<h4>Capacity Comparison:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            categories = ['Theoretical Capacity', 'Actual Capacity']
            values = [theoretical_capacity, actual_capacity]
            colors = ['green', 'blue']

            bars = ax.bar(categories, values, color=colors, width=0.6)

            # Add efficiency indicator
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{height:.1f} kg/h', ha='center', va='bottom')

            # Add efficiency text
            if theoretical_capacity > 0:
                ax.annotate(f'Efficiency: {conveying_efficiency:.1f}%',
                            xy=(0.5, max(values) * 0.5),
                            xytext=(0.5, max(values) * 0.7),
                            textcoords='data',
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

            ax.set_ylabel('Capacity (kg/h)')
            ax.set_title('Theoretical vs. Actual Capacity Comparison')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Create factors affecting efficiency visualization
            st.markdown("<h4>Factors Affecting Conveyor Efficiency:</h4>", unsafe_allow_html=True)

            # Create a radar chart to visualize factors
            factors = ['Belt Tension', 'Material Loading', 'Belt Alignment',
                       'Idler Condition', 'Drive Efficiency', 'Material Properties']

            # Create example values - in a real app, these could be inputs
            factor_values = [0.85, 0.9, 0.95, 0.8, 0.85, 0.9]

            # Number of variables
            N = len(factors)

            # Create angles for each factor
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop

            # Add the values for the complete loop
            factor_values += factor_values[:1]

            # Create the plot
            fig3 = plt.figure(figsize=(10, 8))
            ax3 = fig3.add_subplot(111, polar=True)

            # Draw the polygon and fill it
            ax3.plot(angles, factor_values, 'o-', linewidth=2)
            ax3.fill(angles, factor_values, alpha=0.25)

            # Set the labels
            ax3.set_thetagrids(np.degrees(angles[:-1]), factors)

            # Draw axis lines for each angle and label
            ax3.set_ylim(0, 1)
            ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax3.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax3.grid(True)

            ax3.set_title('Factors Affecting Conveyor Efficiency', size=15, pad=20)

            plt.tight_layout()
            st.pyplot(fig3)

            # Summary of results and recommendations
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### Summary and Recommendations")

            st.markdown(f"""
            The belt conveyor has a theoretical capacity of **{theoretical_capacity:.2f} kg/h** and an actual capacity of **{actual_capacity:.2f} kg/h**, 
            resulting in a conveying efficiency of **{conveying_efficiency:.2f}%**.

            **Recommendations to improve efficiency:**

            1. **Belt Tension**: Ensure proper tensioning to prevent slippage
            2. **Material Loading**: Optimize the feed rate and position
            3. **Belt Alignment**: Check and adjust tracking to prevent material spillage
            4. **Idler Maintenance**: Replace worn or damaged idlers
            5. **Drive System**: Ensure proper power transmission
            6. **Material Distribution**: Ensure even distribution across the belt width
            """)

            st.markdown("</div>", unsafe_allow_html=True)



# Bucket Conveyor Evaluation Module
elif page == "Bucket Conveyor Evaluation":
    st.markdown(
        "<h2 class='sub-header'>Determination of Capacity of a Bucket Conveyor and its Performance Evaluation</h2>",
        unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This module helps determine the capacity and evaluate the performance of bucket elevators (vertical conveyors).
    You can input bucket elevator parameters and experimental data to calculate theoretical capacity, actual capacity, 
    and conveying efficiency.
    </div>
    """, unsafe_allow_html=True)

    # Theory explanation
    with st.expander("Theory and Concepts", expanded=False):
        st.markdown("""
        ### Bucket Elevator Theory

        Bucket elevators (vertical conveyors) are the only equipment used in material handling to convey material in a vertical direction.
        They consist of a series of buckets or cups arranged on an endless belt moving between two flat pulleys.

        #### Working Principle:

        - Buckets/cups fill with grain material during upward movement from the hopper
        - At the top, the material is discharged due to centrifugal force
        - For proper discharge, the weight of material in each bucket should equal the centrifugal force created by rotation

        #### Key Components:

        - Buckets/cups (typically parabolic or rectangular in cross-section)
        - Endless belt or chain
        - Head pulley (top pulley) and boot pulley (bottom pulley)
        - Feed hopper and discharge outlet
        - Drive mechanism (typically at the head pulley)
        - Casing/housing for safety and dust control

        #### Physical Principles:

        **Centrifugal Force Calculation:**
        <div class='formula'>Fc = WV²/(gR)</div>

        Where:
        - W = Weight of material in one cup (kg)
        - V = Speed of belt (m/min)
        - g = Acceleration due to gravity (9.81 m/s²)
        - R = Radius of head pulley (cm)

        **For proper discharge**, the weight of material equals the centrifugal force:
        <div class='formula'>W = WV²/(gR)</div>

        **Speed of Belt:**
        <div class='formula'>V = (gR)½</div>
        <div class='formula'>V = πDN = 2πRN</div>

        **Optimum Speed (RPM):**
        <div class='formula'>N = [(1/2π)/(gR)½]</div>

        #### Capacity Calculations:

        **Number of cups per meter:**
        <div class='formula'>n = 100/s</div>
        Where s = spacing between cups (cm)

        **Material weight per meter of belt:**
        <div class='formula'>W₁ₘ = (100/s) × (vρ/10⁶) kg/m</div>
        Where:
        - v = volume of each cup (cm³)
        - ρ = bulk density of material (kg/m³)

        **Theoretical Capacity:**
        <div class='formula'>Q theoretical = (6ρVv)/(s×10³) kg/h</div>

        **Actual Capacity:**
        <div class='formula'>Q actual = 60W/T kg/h</div>
        Where:
        - W = Total material conveyed (kg)
        - T = Time taken (min)

        **Conveying Efficiency:**
        <div class='formula'>η Conveying = (Q actual/Q theo) × 100%</div>
        """, unsafe_allow_html=True)

        # Add diagrams
        st.image("https://via.placeholder.com/800x400.png?text=Bucket+Elevator+Components+Diagram",
                 caption="Fig. Components of a bucket elevator system")

        col1, col2 = st.columns(2)
        with col1:
            st.image("https://via.placeholder.com/400x300.png?text=Bucket+Cup+Design",
                     caption="Typical bucket/cup design")
        with col2:
            st.image("https://via.placeholder.com/400x300.png?text=Material+Discharge+Diagram",
                     caption="Material discharge via centrifugal force")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Bulk Density Measurement", "Design & Capacity Calculator", "Performance Evaluation"])

    with tab1:
        st.markdown("<h3 class='section-header'>Bulk Density Measurement</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section calculates the bulk density of the material to be conveyed, which is an essential 
        parameter for determining the bucket elevator capacity.
        """)

        with st.form("bucket_bulk_density_form"):
            st.markdown("### Sample Information")

            material_type = st.text_input("Material Type", "Wheat")

            st.markdown("### Bulk Density Measurements")

            col1, col2, col3 = st.columns(3)

            with col1:
                container_volume1 = st.number_input("Container Volume - Sample 1 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0, key="bke_vol1")
                container_mass1 = st.number_input("Container Mass - Sample 1 (g)", min_value=0.0, value=200.0, step=1.0,
                                                  key="bke_mass1")
                total_mass1 = st.number_input("Container + Sample Mass - Sample 1 (g)", min_value=0.0, value=1000.0,
                                              step=1.0, key="bke_total1")

            with col2:
                container_volume2 = st.number_input("Container Volume - Sample 2 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0, key="bke_vol2")
                container_mass2 = st.number_input("Container Mass - Sample 2 (g)", min_value=0.0, value=200.0, step=1.0,
                                                  key="bke_mass2")
                total_mass2 = st.number_input("Container + Sample Mass - Sample 2 (g)", min_value=0.0, value=980.0,
                                              step=1.0, key="bke_total2")

            with col3:
                container_volume3 = st.number_input("Container Volume - Sample 3 (cc)", min_value=1.0, value=1000.0,
                                                    step=1.0, key="bke_vol3")
                container_mass3 = st.number_input("Container Mass - Sample 3 (g)", min_value=0.0, value=200.0, step=1.0,
                                                  key="bke_mass3")
                total_mass3 = st.number_input("Container + Sample Mass - Sample 3 (g)", min_value=0.0, value=1020.0,
                                              step=1.0, key="bke_total3")

            calculate_density_button = st.form_submit_button("Calculate Bulk Density")

        if calculate_density_button:
            # Calculate sample masses
            sample_mass1 = total_mass1 - container_mass1
            sample_mass2 = total_mass2 - container_mass2
            sample_mass3 = total_mass3 - container_mass3

            # Calculate bulk densities
            bulk_density1 = sample_mass1 / container_volume1  # g/cc
            bulk_density2 = sample_mass2 / container_volume2  # g/cc
            bulk_density3 = sample_mass3 / container_volume3  # g/cc

            # Calculate average bulk density
            avg_bulk_density_gcc = (bulk_density1 + bulk_density2 + bulk_density3) / 3
            avg_bulk_density_kgm3 = avg_bulk_density_gcc * 1000  # Convert g/cc to kg/m³

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Bulk Density Results")

            # Create a table of results - without formatting to avoid errors
            results_data = {
                "Sample": ["Sample 1", "Sample 2", "Sample 3", "Average"],
                "Sample Mass (g)": [sample_mass1, sample_mass2, sample_mass3, None],
                "Volume (cc)": [container_volume1, container_volume2, container_volume3, None],
                "Bulk Density (g/cc)": [bulk_density1, bulk_density2, bulk_density3, avg_bulk_density_gcc],
                "Bulk Density (kg/m³)": [bulk_density1 * 1000, bulk_density2 * 1000, bulk_density3 * 1000,
                                         avg_bulk_density_kgm3]
            }

            results_df = pd.DataFrame(results_data)

            # Display the dataframe without formatting
            st.dataframe(results_df)

            st.markdown(
                f"**The average bulk density of {material_type} is {avg_bulk_density_gcc:.4f} g/cc or {avg_bulk_density_kgm3:.1f} kg/m³**")

            st.markdown("</div>", unsafe_allow_html=True)

            # Store the bulk density in session state for use in other tabs
            st.session_state.be_bulk_density = avg_bulk_density_kgm3
            st.session_state.be_material_type = material_type

    with tab2:
        st.markdown("<h3 class='section-header'>Bucket Elevator Design & Capacity Calculator</h3>",
                    unsafe_allow_html=True)

        st.markdown("""
        This calculator determines the theoretical capacity of a bucket elevator and optimal operating parameters
        based on physical principles involving centrifugal forces.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Bucket & Pulley Parameters")

            bucket_volume = st.number_input("Volume of Each Bucket/Cup (v) [cm³]", min_value=1.0, value=500.0,
                                            step=10.0)
            head_pulley_diameter = st.number_input("Diameter of Head Pulley (D) [cm]", min_value=1.0, value=30.0,
                                                   step=0.5)
            bucket_spacing = st.number_input("Spacing Between Buckets (s) [cm]", min_value=1.0, value=12.0, step=0.5)
            elevator_height = st.number_input("Height of Elevator (L) [m]", min_value=0.1, value=3.0, step=0.1)

        with col2:
            st.markdown("### Operating Parameters")

            # Calculate optimal pulley speed based on centrifugal force
            head_pulley_radius = head_pulley_diameter / 2
            g = 9.81  # m/s²

            # Calculate optimal speed N = [(1/2π)/(gR)^(1/2)]
            # Convert radius from cm to m for calculation
            radius_m = head_pulley_radius / 100
            optimal_rpm = (1 / (2 * np.pi)) * (1 / np.sqrt(g * radius_m))
            optimal_rpm *= 60  # Convert from rps to rpm

            pulley_speed = st.number_input("Speed of Pulley (N) [rpm]",
                                           min_value=1.0, max_value=500.0, value=optimal_rpm, step=1.0)

            # Calculate belt speed from pulley speed: V = πDN/100
            belt_speed = (np.pi * head_pulley_diameter * pulley_speed) / 100  # m/min

            st.markdown(f"**Calculated Belt Speed (V):** {belt_speed:.2f} m/min")

            # Get material density from session state or allow manual input
            if 'be_bulk_density' in st.session_state:
                default_density = st.session_state.be_bulk_density
                material_name = st.session_state.be_material_type
            else:
                default_density = 750.0
                material_name = "material"

            material_density = st.number_input(f"Bulk Density of {material_name} (ρ) [kg/m³]",
                                               min_value=100.0, max_value=3000.0, value=default_density, step=10.0)

        if st.button("Calculate Theoretical Capacity & Design Parameters"):
            # Calculate number of buckets per meter of belt
            buckets_per_meter = 100 / bucket_spacing

            # Calculate weight of material per meter of belt
            material_weight_per_meter = (100 / bucket_spacing) * (bucket_volume * material_density / 1e6)  # kg/m

            # Calculate theoretical capacity
            # Q = (6ρVv)/(s×10³) kg/h
            theoretical_capacity = (6 * material_density * belt_speed * bucket_volume) / (bucket_spacing * 1e3)  # kg/h

            # Calculate centrifugal force for proper discharge
            # First, calculate the expected weight per bucket
            weight_per_bucket = bucket_volume * material_density / 1e6  # kg

            # Calculate centrifugal force: Fc = WV²/(gR)
            # Convert radius to m and belt speed to m/s
            radius_m = head_pulley_radius / 100
            belt_speed_ms = belt_speed / 60
            centrifugal_force = (weight_per_bucket * belt_speed_ms ** 2) / (g * radius_m)  # kg (force)

            # Calculate discharge effectiveness ratio (should be close to 1 for optimal discharge)
            discharge_ratio = centrifugal_force / weight_per_bucket

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Design & Capacity Results")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Number of Buckets per Meter:** {buckets_per_meter:.2f}")
                st.markdown(f"**Material Weight per Meter:** {material_weight_per_meter:.3f} kg/m")
                st.markdown(f"**Weight per Bucket:** {weight_per_bucket * 1000:.2f} g")

            with col2:
                st.markdown(f"**Centrifugal Force at Discharge:** {centrifugal_force * 1000:.2f} g")
                st.markdown(f"**Discharge Effectiveness Ratio:** {discharge_ratio:.2f}")
                st.markdown(f"**Theoretical Capacity:** {theoretical_capacity:.2f} kg/h")

            if discharge_ratio < 0.9:
                st.warning(
                    "Discharge ratio is low. Increase belt speed or decrease pulley radius for better discharge.")
            elif discharge_ratio > 1.1:
                st.warning(
                    "Discharge ratio is high. Decrease belt speed or increase pulley radius to prevent premature discharge.")
            else:
                st.success("Discharge ratio is optimal for good material discharge.")

            st.markdown("</div>", unsafe_allow_html=True)

            # Visualizations
            st.markdown("<h4>Bucket Elevator Parameters Visualization:</h4>", unsafe_allow_html=True)

            # Create a diagram showing the elevator and forces
            fig, ax = plt.subplots(figsize=(10, 8))

            # Define coordinates for a schematic diagram
            # Head pulley
            head_x = 50
            head_y = 80
            head_radius = 10

            # Boot pulley
            boot_x = 50
            boot_y = 20
            boot_radius = 8

            # Draw head pulley
            head_pulley = plt.Circle((head_x, head_y), head_radius, fill=False, color='black')
            ax.add_patch(head_pulley)

            # Draw boot pulley
            boot_pulley = plt.Circle((boot_x, boot_y), boot_radius, fill=False, color='black')
            ax.add_patch(boot_pulley)

            # Draw elevator casing
            left_casing = plt.Line2D([head_x - 15, boot_x - 15], [head_y, boot_y], color='black')
            right_casing = plt.Line2D([head_x + 15, boot_x + 15], [head_y, boot_y], color='black')
            ax.add_artist(left_casing)
            ax.add_artist(right_casing)

            # Draw belts
            left_belt = plt.Line2D([head_x - head_radius, boot_x - boot_radius], [head_y, boot_y], color='green')
            right_belt = plt.Line2D([head_x + head_radius, boot_x + boot_radius], [head_y, boot_y], color='green')
            ax.add_artist(left_belt)
            ax.add_artist(right_belt)

            # Draw buckets at intervals
            num_buckets = int(((head_y - boot_y) / 100) * buckets_per_meter)
            bucket_spacing_px = (head_y - boot_y) / (num_buckets + 1)

            for i in range(1, num_buckets + 1):
                y_pos = boot_y + i * bucket_spacing_px

                # Draw bucket on right side (up)
                bucket_width = 8
                bucket_height = 6
                rect = plt.Rectangle((head_x + head_radius - bucket_width / 2, y_pos - bucket_height / 2),
                                     bucket_width, bucket_height, angle=0, color='orange', alpha=0.7)
                ax.add_patch(rect)

                # Draw bucket on left side (down) for empty buckets
                if i % 3 == 0:  # Draw fewer buckets on return side for clarity
                    rect = plt.Rectangle((head_x - head_radius - bucket_width / 2, y_pos - bucket_height / 2),
                                         bucket_width, bucket_height, angle=0, color='khaki', alpha=0.5)
                    ax.add_patch(rect)

            # Draw discharge area
            discharge_arrow = plt.arrow(head_x + 20, head_y, 15, -5, head_width=3, head_length=3,
                                        fc='brown', ec='brown')
            ax.add_artist(discharge_arrow)

            # Draw feed area
            feed_arrow = plt.arrow(boot_x - 20, boot_y + 10, 15, 5, head_width=3, head_length=3,
                                   fc='brown', ec='brown')
            ax.add_artist(feed_arrow)

            # Add labels
            ax.text(head_x, head_y + head_radius + 5, 'Head Pulley', ha='center')
            ax.text(boot_x, boot_y - boot_radius - 5, 'Boot Pulley', ha='center')
            ax.text(head_x + 35, head_y, 'Discharge', ha='center')
            ax.text(boot_x - 35, boot_y + 10, 'Feed', ha='center')
            ax.text(head_x + 25, (head_y + boot_y) / 2, 'Loaded Buckets', ha='center')
            ax.text(head_x - 25, (head_y + boot_y) / 2, 'Empty Buckets', ha='center')

            # Add force diagram at discharge point
            # Draw centrifugal force arrow
            cf_arrow = plt.arrow(head_x + 5, head_y, 15, 0, head_width=2, head_length=3,
                                 fc='red', ec='red')
            ax.add_artist(cf_arrow)
            ax.text(head_x + 15, head_y + 3, 'Fc', color='red', ha='center')

            # Draw gravity force arrow
            g_arrow = plt.arrow(head_x, head_y - 5, 0, -8, head_width=2, head_length=3,
                                fc='blue', ec='blue')
            ax.add_artist(g_arrow)
            ax.text(head_x - 3, head_y - 10, 'g', color='blue', ha='center')

            # Set limits and remove axes
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal')
            ax.axis('off')

            # Add title and capacity information
            plt.title('Bucket Elevator Schematic and Force Diagram', fontsize=14)
            plt.figtext(0.5, 0.02, f"Theoretical Capacity: {theoretical_capacity:.2f} kg/h",
                        ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))

            st.pyplot(fig)

            # Create another visualization showing speed vs. capacity
            st.markdown("<h4>Relationship Between Belt Speed and Capacity:</h4>", unsafe_allow_html=True)

            # Create a range of speeds
            speeds = np.linspace(belt_speed * 0.5, belt_speed * 1.5, 100)
            capacities = [(6 * material_density * s * bucket_volume) / (bucket_spacing * 1e3) for s in speeds]

            # Create discharge ratios
            discharge_ratios = []
            for s in speeds:
                # Convert to m/s for calculation
                s_ms = s / 60
                cf = (weight_per_bucket * s_ms ** 2) / (g * radius_m)
                discharge_ratios.append(cf / weight_per_bucket)

            # Create the plot
            fig2, ax1 = plt.subplots(figsize=(10, 6))

            # Plot capacity line
            color = 'tab:blue'
            ax1.set_xlabel('Belt Speed (m/min)')
            ax1.set_ylabel('Capacity (kg/h)', color=color)
            ax1.plot(speeds, capacities, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Create second y-axis for discharge ratio
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Discharge Ratio', color=color)
            ax2.plot(speeds, discharge_ratios, color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

            # Add optimal zone shading for discharge ratio 0.9-1.1
            ideal_min = 0.9
            ideal_max = 1.1
            ax2.axhspan(ideal_min, ideal_max, alpha=0.2, color='green')

            # Add vertical line for current speed
            plt.axvline(x=belt_speed, color='green', linestyle='-', alpha=0.7,
                        label=f'Current Speed: {belt_speed:.1f} m/min')

            # Mark the current values
            ax1.plot(belt_speed, theoretical_capacity, 'bo', markersize=8)
            ax2.plot(belt_speed, discharge_ratio, 'ro', markersize=8)

            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            fig2.tight_layout()
            plt.title('Belt Speed vs. Capacity and Discharge Ratio')
            st.pyplot(fig2)

            # Store values in session state for use in performance tab
            st.session_state.be_theoretical_capacity = theoretical_capacity
            st.session_state.be_bucket_volume = bucket_volume
            st.session_state.be_belt_speed = belt_speed
            st.session_state.be_design_parameters = {
                'head_pulley_diameter': head_pulley_diameter,
                'bucket_spacing': bucket_spacing,
                'buckets_per_meter': buckets_per_meter,
                'material_weight_per_meter': material_weight_per_meter,
                'discharge_ratio': discharge_ratio
            }

    with tab3:
        st.markdown("<h3 class='section-header'>Performance Evaluation</h3>", unsafe_allow_html=True)

        st.markdown("""
        This section evaluates the actual performance of the bucket elevator by comparing 
        the theoretical capacity with the actual capacity measured experimentally.
        """)

        # Create a form for the actual capacity measurement
        with st.form("be_performance_form"):
            st.markdown("### Actual Capacity Measurement")

            col1, col2 = st.columns(2)

            with col1:
                material_mass = st.number_input("Mass of Material Conveyed (W) [kg]", min_value=0.1, value=10.0,
                                                step=0.1, key="be_mass")
                conveying_time = st.number_input("Time Taken to Convey the Material (T) [min]", min_value=0.1,
                                                 value=1.0, step=0.1, key="be_time")

            with col2:
                st.markdown("### Reference Parameters")

                # Display theoretical capacity if available
                if 'be_theoretical_capacity' in st.session_state:
                    st.info(f"Theoretical Capacity: {st.session_state.be_theoretical_capacity:.2f} kg/h")
                else:
                    st.warning("Calculate theoretical capacity first in the previous tab")

            # Optional power measurement
            st.markdown("### Power Consumption (Optional)")
            col1, col2 = st.columns(2)
            with col1:
                no_load_power = st.number_input("No-Load Power (W)", min_value=0.0, value=100.0, step=10.0)
            with col2:
                loaded_power = st.number_input("Loaded Power (W)", min_value=0.0, value=250.0, step=10.0)

            evaluate_button = st.form_submit_button("Evaluate Performance")

        if evaluate_button:
            # Calculate actual capacity
            actual_capacity = (material_mass / conveying_time) * 60  # kg/h

            # Get theoretical capacity from session state or use a default
            if 'be_theoretical_capacity' in st.session_state:
                theoretical_capacity = st.session_state.be_theoretical_capacity
            else:
                theoretical_capacity = 0
                st.warning("Theoretical capacity not found. Please calculate it in the previous tab.")

            # Calculate conveying efficiency
            if theoretical_capacity > 0:
                conveying_efficiency = (actual_capacity / theoretical_capacity) * 100  # percentage
            else:
                conveying_efficiency = 0

            # Calculate power efficiency if data is provided
            power_difference = loaded_power - no_load_power
            if power_difference > 0:
                # Calculate energy used per kg of material (Wh/kg)
                energy_per_kg = (power_difference * conveying_time / 60) / material_mass  # Wh/kg
            else:
                energy_per_kg = 0

            # Display results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.markdown("### Performance Evaluation Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Theoretical Capacity:** {theoretical_capacity:.2f} kg/h")
            with col2:
                st.markdown(f"**Actual Capacity:** {actual_capacity:.2f} kg/h")
            with col3:
                st.markdown(f"**Conveying Efficiency:** {conveying_efficiency:.2f}%")

            if power_difference > 0:
                st.markdown(f"**Power Used for Material Lifting:** {power_difference:.2f} W")
                st.markdown(f"**Energy Consumption:** {energy_per_kg:.3f} Wh/kg")

            st.markdown("</div>", unsafe_allow_html=True)

            # Efficiency explanation based on value
            if conveying_efficiency < 50:
                efficiency_comment = "Low efficiency suggests significant material losses or loading/discharge issues."
            elif conveying_efficiency < 80:
                efficiency_comment = "Moderate efficiency. Some improvement possible in bucket filling or discharge."
            elif conveying_efficiency < 95:
                efficiency_comment = "Good efficiency. The bucket elevator is performing well."
            elif conveying_efficiency <= 100:
                efficiency_comment = "Excellent efficiency. The elevator is operating at near-optimal conditions."
            else:
                efficiency_comment = "Efficiency exceeds 100%, which may indicate measurement errors or theoretical capacity underestimation."

            st.markdown(f"**Performance Assessment:** {efficiency_comment}")

            # Create visualization
            st.markdown("<h4>Capacity Comparison:</h4>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            categories = ['Theoretical Capacity', 'Actual Capacity']
            values = [theoretical_capacity, actual_capacity]
            colors = ['green', 'blue']

            bars = ax.bar(categories, values, color=colors, width=0.6)

            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{height:.1f} kg/h', ha='center', va='bottom')

            # Add efficiency text
            if theoretical_capacity > 0:
                ax.annotate(f'Efficiency: {conveying_efficiency:.1f}%',
                            xy=(0.5, max(values) * 0.5),
                            xytext=(0.5, max(values) * 0.7),
                            textcoords='data',
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

            ax.set_ylabel('Capacity (kg/h)')
            ax.set_title('Theoretical vs. Actual Capacity Comparison')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # If we have power data, add a power efficiency chart
            if power_difference > 0:
                st.markdown("<h4>Power Analysis:</h4>", unsafe_allow_html=True)

                fig2, ax2 = plt.subplots(figsize=(10, 6))

                # Create a pie chart showing power distribution
                labels = ['No-Load Power', 'Material Lifting Power']
                sizes = [no_load_power, power_difference]
                explode = (0, 0.1)  # Explode the second slice

                ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90, colors=['lightgray', 'lightblue'])
                ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

                plt.title('Power Distribution in Bucket Elevator Operation')
                st.pyplot(fig2)

                # Add some context about the power efficiency
                total_theoretical_energy = 9.81 * material_mass * st.session_state.get('elevator_height',
                                                                                       3) / 1000  # kWh theoretical lifting energy
                actual_energy_kwh = (power_difference * conveying_time) / (60 * 1000)  # kWh
                if total_theoretical_energy > 0:
                    power_efficiency = (total_theoretical_energy / actual_energy_kwh) * 100
                    st.markdown(f"""
                    **Power Efficiency Analysis:**
                    - Theoretical energy to lift material: {total_theoretical_energy:.4f} kWh
                    - Actual energy used: {actual_energy_kwh:.4f} kWh
                    - Mechanical efficiency: {power_efficiency:.2f}%
                    """)

            # Summary of results and recommendations
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.markdown("### Summary and Recommendations")

            st.markdown(f"""
            The bucket elevator has a theoretical capacity of **{theoretical_capacity:.2f} kg/h** and an actual capacity of **{actual_capacity:.2f} kg/h**, 
            resulting in a conveying efficiency of **{conveying_efficiency:.2f}%**.

            **Recommendations to improve efficiency:**

            1. **Bucket Loading**: Ensure proper feeding rate and position
            2. **Discharge**: Optimize belt speed for proper centrifugal discharge
            3. **Maintenance**: Check for any material spillage or bucket damage
            4. **Belt Tension**: Ensure proper tensioning to prevent slippage
            5. **Power Efficiency**: Minimize mechanical losses in the system
            """)

            st.markdown("</div>", unsafe_allow_html=True)

# Add a footer at the end of all page content
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd;">
    <p>Developed for Protected Cultivation and Secondary Agriculture Course</p>
     
</div>
""", unsafe_allow_html=True)
