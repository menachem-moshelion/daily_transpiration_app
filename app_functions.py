# Import necessary packages
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
import joblib
from scipy.signal import savgol_filter
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import logging
from datetime import timedelta
import re

# Configure logging
logging.basicConfig(filename='logging.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to validate the token
def validate_bearer_token(token):
    pattern = r"^Bearer [a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, token))

def check_soil_sand(df, threshold=4700):
    """
    Determines if the plant is grown in soil or sand based on the initial weight ('s4' column).
    
    Args:
        df (DataFrame): Data containing the 's4' column (weight measurements).
        threshold (int, optional): Threshold to classify between soil and sand. Default is 4700.
    
    Returns:
        str: 'sand' if median weight exceeds threshold, 'soil' if below threshold,
             or 'unknown' if there is insufficient data.
    """
    # Check if 's4' exists and has non-NaN data
    if 's4' not in df.columns or df['s4'].iloc[:1440].dropna().empty:
        logging.warning("Insufficient data in the first 3 days of 's4' to determine soil/sand.")
        return 'unknown'
    
    if df['s4'].iloc[:1440].median() >= 10000:
        logging.warning("Large pots, can't infer soil type by pot weight")
        return 'unknown'
    
    # Calculate median and classify
    median_weight = df['s4'].iloc[:1440].median()
    is_sand = median_weight > threshold

    logging.info(f"Median weight: {median_weight}, Classified as: {'sand' if is_sand else 'soil'}")
    
    return 'sand' if is_sand else 'soil'


def fetch_data_from_SPAC(start_date, end_date, authorization, plants_id, exp_id, control_id):
    """
    Fetches raw data from the SPAC API.
    
    Args:
        start_date (datetime): Start date.
        end_date (datetime): End date.
        authorization (str): API token.
        plants_id (str): Plant ID.
        exp_id (str): Experiment ID.
        control_id (str): Control system ID.

    Returns:
        dict: Dictionary containing raw API data or None if request fails.
    """
    logging.info(f"Fetching data for exp: {exp_id}, plant: {plants_id}, control: {control_id}")
    payload={}
    headers = {'Authorization': f'{authorization}'}
    parameters = "s4,wthrsrh,wthrstemp,wthrspar,wthrsvpd"
    daily_parameters = "dt,pnw"

    start_date_f = start_date.strftime('%Y-%m-%dT%H:%M:%S')
    end_date_f = (end_date + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')

    # URLs
    daily_url = f"https://spac.plant-ditech.com/api/data/getData?experimentId={exp_id}&controlSystemId={control_id}&fromDate={start_date_f}&toDate={end_date_f}&plants={plants_id}&params={daily_parameters}"
    continuous_url = f"https://spac.plant-ditech.com/api/data/getData?experimentId={exp_id}&controlSystemId={control_id}&fromDate={start_date_f}&toDate={end_date_f}&plants={plants_id}&params={parameters}"

    try:
        # Fetch Daily Data
        logging.info(f"Fetching daily data from: {daily_url}")
        daily_response = requests.get(daily_url, headers=headers, data=payload)

        if daily_response.status_code == 500:
            logging.critical("One or more of your inputs are incorrect, or the server is down.")
            return None
        if daily_response.status_code != 200:
            logging.error(f"Daily request failed: {daily_response.status_code}")
            return None
        daily_data = daily_response.json()

        # Fetch Continuous Data
        logging.info(f"Fetching continuous data from: {continuous_url}")
        continuous_response = requests.get(continuous_url, headers=headers, data=payload)
        if continuous_response.status_code != 200:
            logging.error(f"Continuous request failed: {continuous_response.status_code}")
            return None
        continuous_data = continuous_response.json()

        return {"daily": daily_data, "continuous": continuous_data}

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error: {req_err}")
        return None

def process_SPAC_data(raw_data, start_date, end_date, plants_id, exp_id, control_id, plant_type, soil_type=None):
    """
    Processes raw SPAC API data and converts it into structured DataFrames.
    
    Args:
        raw_data (dict): Raw API response data.
        start_date (datetime): Start date.
        end_date (datetime): End date.
        exp_id (str): Experiment ID.
        plant_id (str): Plant ID.
        plant_type (str, optional): Plant type.

    Returns:
        tuple: (processed continuous data, processed daily data)
    """
    if not raw_data:
        logging.error("No raw data provided for processing.")
        return None, None
    
    # Daily Data Processing
    daily_group_data = raw_data["daily"]['group1']
    # Create daily values DataFrame
    daily_tr = pd.DataFrame(daily_group_data['data']['dt'], columns=['timestamp', 'dt'])
    plant_weight = pd.DataFrame(daily_group_data['data']['pnw'], columns=['timestamp', 'pnw'])
    merged_daily_data = pd.merge(daily_tr, plant_weight, on='timestamp', how='outer')
    
    if daily_tr.empty or plant_weight.empty:
        logging.warning("Daily data is empty.")
        return None, None
    
    group_data = raw_data["continuous"]['group1']
    # Prepare data for 's4' and 'wsrh'
    s4_data = pd.DataFrame(group_data['data']['s4'], columns=['timestamp', 's4'])
    wsrh_data = pd.DataFrame(group_data['data']['wthrsrh'], columns=['timestamp', 'wsrh'])
    wstemp_data = pd.DataFrame(group_data['data']['wthrstemp'], columns=['timestamp', 'wstemp'])
    wspar_data = pd.DataFrame(group_data['data']['wthrspar'], columns=['timestamp', 'wspar'])
    wsvpd_data = pd.DataFrame(group_data['data']['wthrsvpd'], columns=['timestamp', 'vpd'])
    
    
    # Merging the four datasets with an outer merge
    merged_data = pd.merge(s4_data, wsrh_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wstemp_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wspar_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wsvpd_data, on='timestamp', how='outer')
    
    # Timestamp organizing for further data adding
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], format='%Y-%m-%dT%H:%M:%S')
    merged_data.set_index('timestamp', inplace=True)
    merged_data.sort_index(inplace=True) 
    
    # Add the missing timestamp check and insertion after sorting the index
    expected_interval = timedelta(minutes=3)
    all_timestamps = pd.date_range(start=start_date, end=end_date, freq=expected_interval)
    merged_data = merged_data.reindex(all_timestamps, fill_value=np.NaN)
    
    # add constent info
    merged_data['control_id'] = control_id
    merged_data['exp_ID'] = exp_id
    merged_data['plant_ID'] = group_data['plants'][0]
    merged_data['plant_type'] = plant_type
    if soil_type:
        merged_data['soil_sand'] = soil_type
    else:
        merged_data['soil_sand'] = check_soil_sand(merged_data, threshold=4700)
        
    merged_data = merged_data.reset_index() # Reset index so timestamp is in the table

    logging.info(f"got the data of plant {plants_id}, exp {exp_id} from Date:{start_date} to {end_date}")
    logging.info(f"df head: \n{merged_data.head(3)}")
    return merged_data, merged_daily_data
    

def get_daily(plant_df, plant_daily_data, control_id):
    """
    Aggregates daily data based on light exposure, weight changes, and other plant parameters.
    
    Args:
    plant_df (DataFrame): DataFrame containing the raw plant data with 3-minute intervals.
    plant_daily_data (DataFrame): External daily data to be merged with.
    control_id (str or int): Control system ID.

    Returns:
    DataFrame: The merged daily data, including light hours, daily means, and plant information.
    """
    
    # Ensure the index is a datetime
    plant_df['timestamp'] = pd.to_datetime(plant_df['index'])

    # Set the timestamp as the index for grouping
    plant_df = plant_df.set_index('timestamp')
    
    # Filter data where there's light (wspar is not 0) #Daily - light time.
    data_in_light = plant_df[plant_df['wspar'] != 0]

    # Calculate daily means for specified columns
    daily_means = data_in_light[['wstemp', 'wsrh', 'wspar','vpd']].resample('D').mean()

    # Take the first value for the specified columns
    daily_first = plant_df[['exp_ID', 'plant_ID', 'plant_type', 'soil_sand']].resample('D').first()

    # Calculate light hours (3-minute intervals; divide by 20 to get hours)
    light_hours = plant_df['wspar'].gt(0).resample('D').sum() / 20 
    light_hours = light_hours.to_frame(name='light_hours') 
    
    # Calculate DLI using the formula
    daily_dli = plant_df['wspar'].resample('D').sum() / 480 * 86400 / 1000000
    daily_dli = daily_dli.to_frame(name='DLI')

    # Combine all the calculated daily data into a single DataFrame
    daily_data = pd.concat([daily_means, daily_first, light_hours, daily_dli], axis=1) #daily_vpd_sum
    
    # Add control_id to the daily data
    daily_data['control_id'] = control_id
    
    # Ensure the timestamp column is a datetime in the plant_daily_data
    plant_daily_data['timestamp'] = pd.to_datetime(plant_daily_data['timestamp'])

    # Set the timestamp column as the index
    plant_daily_data.set_index('timestamp', inplace=True)

    # Merge the calculated daily data with the plant_daily_data on timestamp
    merged_data = daily_data.join(plant_daily_data, how='outer')
    merged_data = merged_data.reset_index() # Reset the index to make timestamp a column again
    logging.info(f"daily df head: \n{merged_data.head(3)}")

    return merged_data

def process_plant_weight(data):

    # Sort by 'plantID' and then by date
    data.sort_values(by=['plant_ID', 'timestamp'], inplace=True)

    # Group by plantID
    grouped = data.groupby('plant_ID', group_keys=False)


    # Apply the smoothing function and process the group
    def smooth_data(data, window_length=9, polyorder=2):
        # Check if the data length is sufficient for the given window_length
        # If not, adjust the window_length to the next smallest odd number
        if len(data) < window_length:
            window_length = len(data) - (len(data) % 2) - 1  # Decrease to the nearest odd number smaller than the length of data
        if window_length > 2:  # window_length must be at least 3 for the savgol_filter to work
            return savgol_filter(data, window_length, polyorder, mode='nearest')  # Apply Savitzky-Golay filter
        else:
            return data  # If data is too short, return it unchanged

    def process_group(group):
        group = group.copy()
        # Replace values under 3 with NaN
        group['plant_weight_process'] = group['plant_weight'].apply(lambda x: x if x >= 3 else np.nan)
        
        # If the first 4 values are NaN, set a base start of 10 g (shift_value)
        if group['plant_weight_process'].iloc[:4].isnull().all() or group['plant_weight_process'].max() > 1500:

            if group['plant_weight_process'].max() > 1500:
                st.session_state["weight_auto_reset"] = True
                st.session_state["original_high_weight"] = float(
                    group['plant_weight_process'].iloc[0]
                )

            # Shift the series to start from 10g
            first_valid_index = group['plant_weight'].first_valid_index() #sometimes starts with Nan so we need this

            group['plant_weight_process'] = group['plant_weight'] - group.loc[first_valid_index, 'plant_weight'] + 10
        
        # Apply cumulative maximum to ensure increasing trend
        group['plant_weight_process'] = group['plant_weight_process'].cummax()
        
        # Interpolate missing values linearly
        group['plant_weight_process'] = group['plant_weight_process'].interpolate(method='linear', limit_direction='both')

        # Smooth the data
        group['plant_weight_process'] = smooth_data(group['plant_weight_process']) 
        return group

    # Apply processing to each group
    df_processed = grouped.apply(process_group)

    # Merge processed data back to original DataFrame
    data['plant_weight_process'] = df_processed['plant_weight_process']

    return data

def get_daily_data_from_SPAC(start_date, end_date, authorization, plant_id, exp_id, control_id, plant_type, soil_type):
    """
    Fetches and processes daily aggregated data from SPAC
    Returns:
        selected_df (DataFrame): Processed DataFrame with relevant columns.
    """
    raw_data = fetch_data_from_SPAC(start_date, end_date, authorization, plant_id, exp_id, control_id)

    if raw_data is None:
        logging.error("Failed to retrieve raw data.")
        return None

    plant_df, plant_daily_data = process_SPAC_data(raw_data, start_date, end_date, plant_id, exp_id, control_id, plant_type, soil_type)
    if plant_df is None or plant_daily_data is None:
        logging.error("Processing step failed.")
        return None

    plant_full_daily_data = get_daily(plant_df, plant_daily_data, control_id)

    plant_full_daily_data['encoded_plant'] = (plant_full_daily_data['plant_type']
        .map({'cereal': 1, 'tomato': 0}).astype(float))
    plant_full_daily_data['encoded_soil'] = (plant_full_daily_data['soil_sand']
        .map({'sand': 0, 'soil': 1}).astype(float))
    
    plant_full_daily_data.rename(columns={"wsrh": "RH", "wstemp": "Temp", "dt" : "Transpiration", "pnw": "plant_weight"}, inplace=True)

    plant_full_daily_data = process_plant_weight(plant_full_daily_data)
    
    logging.info("Selecting relevant columns from DataFrame.")
    columns = ['vpd', 'Temp', 'RH', 'DLI', 'Transpiration', 'encoded_plant', 'encoded_soil','plant_weight_process']
    selected_df = plant_full_daily_data[columns].copy()

    logging.info("Final dataset prepared successfully.")
    return selected_df



def clean_data(df):
    """
    Remove rows with NaN values to ensure clean data for model evaluation.
    """
    df_cleaned = df.dropna()
    logging.info(f"Dropped {len(df) - len(df_cleaned)} rows with NaN values.")
    return df_cleaned

#################### testing models functions ###############

def adjust_plant_weight(df):
    """
    Allow user to adjust the first plant weight value and modify the column accordingly.
    """
    if df is None or df.empty:
        st.error("No data available to adjust plant weight.")
        return None

    first_weight = df['plant_weight_process'].iloc[0]

    if "weight_adjusted" not in st.session_state:
        st.session_state["weight_adjusted"] = False
        st.session_state["custom_adjustment"] = first_weight  # Default to original weight

    # Display the first weight
    st.write(f"**First plant weight:** {first_weight} g")

    if first_weight > 20 and not st.session_state["weight_adjusted"]:
        st.warning("⚠️ The plant's net weight is very high! \n\nThe usual inital weight of a planted tomato or cereal plant is between 8-10g. \n\nPerhaps you weighed your plant along with the bulk of soil?")
        recommended_adjustment = first_weight - 10  # Targeting an initial weight of ~10g

        # Layout with side-by-side buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"✅ Recommended:\n subtract {recommended_adjustment:.2f}g"):
                df['plant_weight_process'] -= recommended_adjustment
                st.session_state["weight_adjusted"] = True
                st.success("✅ Weight adjusted! The initial weight is now approximately 10g.")

        with col2:
            user_input = st.number_input("Enter correct first plant weight:", value=st.session_state["custom_adjustment"])
            if st.button("🔄 Apply my custom adjustment"):
                adjustment = first_weight - user_input
                df['plant_weight_process'] -= adjustment
                st.session_state["weight_adjusted"] = True
                st.success(f"✅ Plant weight column adjusted by {adjustment}g.")

        with col3:
            if st.button("❌ Keep original values"):
                st.session_state["weight_adjusted"] = True
                st.info("ℹ No changes made to the data.")

    return df


def evaluate_and_compare_models(models, X_test, y_test, scaler=None):
    """
    Evaluate and compare regression models based on performance metrics and statistical tests.

    Parameters:
    -----------
    models : list
        A list of models that have been fitted. Each model in the list can be a tuple containing the model instance and a string label for the model.
    X_test : array-like or sparse matrix
        Test feature data.
    y_test : array-like
        True labels for the test data.
    scaler : object, optional
        Scaler object for scaling test data if needed, default is None.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing performance metrics and p-values from statistical tests for all model pairs.
    """


    logging.info(f"comparing modles performance")
    def evaluate_model(model, X_test, y_test, is_nn=False, scaler=None):
        if is_nn and scaler is not None:
            X_used = scaler.transform(X_test)
        else:
            X_used = X_test
        y_pred = model.predict(X_used)
        y_pred = np.ravel(y_pred)  # Flatten predictions to 1D ##

        # Flatten y_test to ensure it's 1D as well ##
        y_test = np.ravel(y_test) ##

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        relative_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mean_abs_residual = np.mean(abs(y_test - y_pred))
        mean_abs_residual_percent = np.mean((abs(y_test - y_pred)) / y_test) * 100

        return r2, rmse, mae, relative_error, mean_abs_residual, mean_abs_residual_percent

    model_metrics = {}
    
    for model, model_name in models:
        is_nn = isinstance(model, Sequential)
        model_metrics[model_name] = evaluate_model(model, X_test, y_test, is_nn, scaler)

    # Create DataFrame for results
    metrics_df = pd.DataFrame(model_metrics.values(), 
                              columns=['R2', 'RMSE', 'MAE', 'Relative Error (%)', 'Mean Abs Residual (g)', 'Mean Abs Residual (%)'],
                              index=model_metrics.keys())
    
    return metrics_df

def y_predictions_df(models, X, y_true, scaler=None):
    """
    Generate a DataFrame with predictions from multiple models, scaling input features for the neural network model.

    Parameters:
    -----------
    models : list
        A list of models that have been fitted. Each model in the list can be a tuple containing the model instance and a string label for the model.
    X : pandas.DataFrame
        The features data used for making predictions.
    y_true : array-like
        The true target values.
    scaler : object, optional
        Scaler object to use for scaling data for neural network models.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the true values, the input features and the predictions from each model.
    """
    pred_df = pd.DataFrame({'Actual': y_true})
    for model, label in models:
        if label == 'Neural Network' and scaler:
            X_transformed = scaler.transform(X)
            y_pred = model.predict(X_transformed)
        else:
            y_pred = model.predict(X)
        pred_df[label] = y_pred
    
    # Add input features to the DataFrame
    pred_df = pd.concat([X, pred_df], axis=1)
    
    return pred_df

def correlation_chart(y_actual, y_pred, model_name):
    """
    Generate a correlation plot between actual and predicted transpiration values.
    """
    logging.info(f"Generating correlation plot for {model_name}")
    df_corr = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=df_corr, alpha=0.5)
    sns.regplot(x='Actual', y='Predicted', data=df_corr, scatter=False, color='red')
    plt.xlabel("Actual Transpiration")
    plt.ylabel("Predicted Transpiration")
    plt.title(f"Actual vs Predicted Transpiration ({model_name})")
    st.pyplot(plt)
    plt.close()

def load_and_test_models(x, y):
    x.rename(columns={'vpd': 'VPD'}, inplace=True) #to match the column name in the model

    logging.info("Loading models for evaluation.")
    dt_model = joblib.load('models/dt_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    nn_model = load_model('models/nn_model.h5')
    scaler_loaded = joblib.load('models/scaler.pkl')
    
    fitted_models = [(dt_model, 'Decision Tree'), (rf_model, 'Random Forest'), (xgb_model, 'XGBoost'), (nn_model, 'Neural Network')]
    evaluation_df = evaluate_and_compare_models(fitted_models, x, y, scaler_loaded)
    
    # Identify the best model (highest R2 score)
    best_model_name = evaluation_df['R2'].idxmax()
    best_model = next(model for model, name in fitted_models if name == best_model_name)
    logging.info(f"the best model is: {best_model_name}")

    # Generate predictions for correlation analysis
    y_pred = best_model.predict(x)
    y_pred = np.ravel(y_pred)  # Flatten predictions to 1D
    y = np.ravel(y) # Flatten y_test to ensure it's 1D as well
    
    correlation_chart(y, y_pred, best_model_name)
    
    # Check if model performance is poor
    if evaluation_df.loc[best_model_name, 'R2'] < 0.5:
        st.warning("Model performance is low. Please verify the following:")
        st.warning("- Is the plant weight measurement trustable?")
        st.warning("- Is the plant grown in 4L pots?")
        
        current_soil_type = x['encoded_soil'].iloc[0] #map({0: 'sand', 1: 'soil'})
        if current_soil_type==0:
            st.warning("- Is this plant grown in sand (Silica sand grade 20-30)? ")  
        elif current_soil_type==1:
            st.warning("- Is this plant grown in soil (55% peat, 20% tuff and 25% puffed coconut coir fiber)?")

        current_crop_type = x['encoded_plant'].iloc[0] #map({0:'tomato', 1:'cereal'})
        if current_crop_type==0:
            st.warning("- Is this plant actually a tomato (similar to M82)? ")  
        elif current_crop_type==1:
            st.warning("- Is this plant actually a cereal (like wheat or barley)?")
        
    #plot
    # Generate predictions using the y_predictions_df function
    pred_df = y_predictions_df(fitted_models, x, y, scaler_loaded)

    # Plot directly using the wide format
    fig = px.line(
        pred_df, 
        x=pred_df.index,  # Use the index as the x-axis
        y=['Actual', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network'],  # Specify the columns for y-axis
        labels={'value': 'Transpiration', 'index': 'days'},
        title="Actual vs Predicted Transpiration for Different Models"
    )

    # Customize layout
    fig.update_layout(legend_title_text='Model', xaxis_title="Index", yaxis_title="Transpiration")

    return evaluation_df, fig

