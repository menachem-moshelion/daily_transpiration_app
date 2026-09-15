import streamlit as st
from app_functions import validate_bearer_token, get_daily_data_from_SPAC, load_and_test_models, clean_data, adjust_plant_weight
import logging

# Configure logging
logging.basicConfig(filename='logging.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#  in terminal run:
# cd Daily_transpiration\APP
# streamlit run TestModelApp.py

# Define the Streamlit interface
st.title("Model Testing App 🤖🌿")
st.markdown("### Welcome to the Model Testing App!")
st.write(
    "This web application allows users to test our trained models on their own data from the SPAC analytics software.\n\n"
    "To use this app:\n\n"
    "you most have accssess to an experiment at the SPAC analytics\n\n"
    "You sould open the experimnt, and inspect the relevent plant using Analysis--> Graph viewer\n\n"
    "Look at the daily transpiration and the plant net weight to mack sure these parameters are accuarte\n\n"
)
st.warning("⚠️ Note: Your data is not stored or saved in any way.")
st.warning("This app is for testing purposes only. The models are trained on specific data and may not generalize well to all experiments and greenhouses.")

# Info Button with Expander
with st.expander("ℹ️ Where can I find my Control ID, Experiment ID, or Plant ID?"):
    st.write(
        "Your **Control ID**, **Experiment ID**, and **Plant ID** can be found in the SPAC analytics software "
        "when inspecting your experiment using **Analysis** → **Graph viewer**. \n\n"
        "1. Fill in the dates and select the plant you would like to analyze.\n"
        "2. Choose a parameter and click **SHOW GRAPH**.\n"
        "3. The **Control ID** and **Experiment ID** will appear in the URL at the top of the page.\n"
        "4. The **Plant ID** is displayed in parentheses next to the plant name in the filter section.\n\n"
        "See the example screenshot below:"
    )
    st.image("images/ID_where_to_fined.png", caption="SPAC Analytics - Where to find your Control ID, Experiment ID, or Plant ID", use_container_width=True)


# User input for control ID and plant ID as integers
control_id = st.number_input("Enter Control ID", min_value=1, step=1)  # min_value and step restrict the input to integers
exp_id = st.number_input("Enter Experiment ID", min_value=1, step=1)
plant_id = st.number_input("Enter Plant ID", min_value=1, step=1)

start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
plant_type = st.selectbox("Choose Plant Type", ["tomato", "cereal"])
soil_type = st.selectbox("Choose Soil Type", ["sand", "soil"])

# Checkbox for irrigation status
irrigation_status = st.checkbox("This plant was well irrigated and did not reach pot limitation between the dates provided")

# User Input Fields
authorization = st.text_input("🔑 Enter your SPAC Authorization Token:", type="password")
with st.expander("ℹ️ How to Find Your SPAC Authorization Token"):
    st.write(
        "To access data from SPAC, you need to provide your **Authorization Token**. "
        "Follow these steps to find it:\n\n"
        "1. **Open SPAC Analytics** and log into your account.\n"
        "2. Press **`Ctrl + Shift + I`** (Windows/Linux) or **`Cmd + Option + I`** (Mac) to open the browser's Developer Tools.\n"
        "3. Navigate to the **Network** tab.\n"
        "4. click **SHOW GRAPH**.\n"
        "5. Click on one of the requests named **`getData?...`**\n"
        "6. Select the **Headers** tab and scroll down to find the field labeled **`Authorization`**.\n"
        "7. Copy the token value (it starts with `Bearer ...`).\n"
        "8. Paste it in the input box below."
    )

    # Embed the instructional video
    video_path = "images/SPAC_authorization.mp4"  # Ensure this file is added to your GitHub repo
    st.video(video_path)

# Check token validity
if authorization:
    if validate_bearer_token(authorization):
        st.success("✅ Valid Authorization Token!")
    else:
        st.error("❌ Invalid Authorization Token. Please ensure it starts with 'Bearer ' and follows the correct format.")

# Button to Fetch Data
if st.button("Get my data"):
    st.write("This might tack some time....")
    # Clear all session states
    st.session_state.clear()

    if not authorization:
        st.error("Please enter your SPAC Authorization Token.")
    
    else:
        try:
            user_data = get_daily_data_from_SPAC(start_date, end_date, authorization, plant_id, exp_id, control_id, plant_type, soil_type)
            if user_data is None:
                st.error("Data retrieval failed. Please check your inputs or try again later.")
            else:
                st.session_state["user_data"] = user_data  # Store data in session state
                st.success("Data successfully retrieved!")
        except Exception as e:
            st.error(f"Failed to retrieve data: {e}")

# If data is available, allow inspection and modification
if "user_data" in st.session_state:
    st.subheader("Inspect & Modify Data")

    if st.session_state.get("weight_auto_reset", False):
        original_weight = st.session_state.get("original_high_weight")

        st.warning(
            f"⚠️ The initial plant weight was unusually high ({original_weight:.1f} g)."
            "Because the recorded weight exceeded 1500 g, the app automatically "
            "recalculated plant biomass so that the initial processed plant weight "
            "is approximately 10 g."
        )
        
    # Apply adjustment function
    adjusted_data = adjust_plant_weight(st.session_state["user_data"])

    # Editable Data Table
    edited_data = st.data_editor(adjusted_data, num_rows="dynamic")
    
    # Store modified data
    st.session_state["edited_data"] = edited_data

    # Correlation Analysis
    st.subheader("TEST MODEL ON MY DATA")
    
    if st.button("Test the Model"):
        st.write("🚀 Running model... ")
        if "edited_data" in st.session_state and not st.session_state["edited_data"].empty:
            model_data = st.session_state["edited_data"]
            st.info("Running model on edited data...")
        elif "user_data" in st.session_state and not st.session_state["user_data"].empty:
            model_data = st.session_state["user_data"]

            st.warning("Running model on unedited data (please edit if needed).")
        else:
            st.error("No data available. Please fetch data first.")
            st.stop()

        model_data = clean_data(model_data)
        X_test = model_data.drop(['Transpiration'], axis=1).astype(float)
        y_test = model_data['Transpiration'].astype(float)
        evaluation_df, plot_y = load_and_test_models(X_test, y_test)
        st.write(evaluation_df)
        # Display the plot in Streamlit
        st.plotly_chart(plot_y)

