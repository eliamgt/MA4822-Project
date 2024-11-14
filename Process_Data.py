import numpy as np

def smooth_data(current_timestamp, data, window_size=5):
    """
    Calculate the moving average for a given data point based on previous values within a specified window.

    Parameters:
    - current_timestamp: The index of the current data point in the time series.
    - data: The list of data points to be smoothed.
    - window_size: The number of data points to include in the moving average (default is 5).

    Returns:
    - smoothed_value: The moving average of the data points within the window.
    """

    # Define the starting index for the window, ensuring it does not go below 0
    start_index = max(0, current_timestamp - window_size + 1)
    # Define the ending index for the window, which includes the current timestamp
    end_index = current_timestamp + 1

    # Calculate the moving average of data within the window
    smoothed_value = sum(data[start_index:end_index]) / (end_index - start_index)
    
    return smoothed_value

def update_temperature(time_stamp, temp_base, modified_temp, action_level_heater, min_limit, max_limit, decay_rate, action_factor):
    """
    Update the modified temperature based on heater action, base temperature, and decay rate.

    Parameters:
    - time_stamp: The current time index.
    - temp_base: List of base temperature values.
    - modified_temp: List of modified temperature values.
    - action_level_heater: Intensity of heater action.
    - min_limit: Minimum temperature limit.
    - max_limit: Maximum temperature limit.
    - decay_rate: Rate at which the modified temperature approaches the base temperature when no action is applied.
    - action_factor: Scaling factor for the impact of heater action on temperature.

    Returns:
    - new_modified_temp: The updated modified temperature at the current time index.
    """

    # Get the base temperature at the current timestamp
    current_temp_base = temp_base[time_stamp]
    # Get the modified temperature from the previous timestamp
    previous_modified_temp = modified_temp[time_stamp-1]
    
    # Scale heater action level to a range of 0 to 1
    intensity = action_level_heater / 10
    
    # If heater is on, increase temperature toward max limit
    if intensity > 0:
        new_modified_temp = previous_modified_temp + (max_limit - previous_modified_temp) * intensity * action_factor
    # If cooling is on, decrease temperature toward min limit
    elif intensity < 0:
        new_modified_temp = previous_modified_temp - (previous_modified_temp - min_limit) * abs(intensity) * action_factor
    else:
        # If no action, decay towards the base temperature
        if previous_modified_temp > current_temp_base:
            new_modified_temp = max(current_temp_base, previous_modified_temp - decay_rate)
        else:
            new_modified_temp = min(current_temp_base, previous_modified_temp + decay_rate)

    return new_modified_temp

def update_humidity(time_stamp, humidity_base, modified_humidity, action_level_dehumidifier, action_level_ventilation, min_limit, decay_rate, action_factor):
    """
    Update the modified humidity based on dehumidifier and ventilation actions, base humidity, and decay rate.

    Parameters:
    - time_stamp: The current time index.
    - humidity_base: List of base humidity values.
    - modified_humidity: List of modified humidity values.
    - action_level_dehumidifier: Intensity of dehumidifier action.
    - action_level_ventilation: Intensity of ventilation action.
    - min_limit: Minimum humidity limit.
    - decay_rate: Rate at which modified humidity approaches the base humidity when no action is applied.
    - action_factor: Scaling factor for the impact of dehumidifier/ventilation on humidity.

    Returns:
    - new_modified_humidity: The updated modified humidity at the current time index.
    """

    # Get the base humidity at the current timestamp
    current_humidity_base = humidity_base[time_stamp]
    # Get the modified humidity from the previous timestamp
    previous_modified_humidity = modified_humidity[time_stamp-1]

    # Combine dehumidifier and ventilation actions into a single intensity measure, scaled to 0-1
    intensity = (action_level_dehumidifier + action_level_ventilation) / 10

    # If dehumidifier or ventilation is on, decrease humidity towards min limit
    if intensity > 0:
        new_modified_humidity = previous_modified_humidity - (previous_modified_humidity - min_limit) * intensity * action_factor
    else:
        # Decay humidity towards base humidity when no action is applied
        new_modified_humidity = previous_modified_humidity + (current_humidity_base - previous_modified_humidity) * decay_rate
    
    return new_modified_humidity

def update_moisture(time_stamp, moisture_base, modified_moisture, action_level_ventilation, min_limit, decay_rate, action_factor):
    """
    Update the modified moisture based on ventilation action, base moisture, and decay rate.

    Parameters:
    - time_stamp: The current time index.
    - moisture_base: List of base moisture values.
    - modified_moisture: List of modified moisture values.
    - action_level_ventilation: Intensity of ventilation action.
    - min_limit: Minimum moisture limit.
    - decay_rate: Rate at which modified moisture approaches the base moisture when no action is applied.
    - action_factor: Scaling factor for the impact of ventilation on moisture.

    Returns:
    - new_modified_moisture: The updated modified moisture at the current time index.
    """

    # Get the base moisture at the current timestamp
    current_moisture_base = moisture_base[time_stamp]
    # Get the modified moisture from the previous timestamp
    previous_modified_moisture = modified_moisture[time_stamp-1]

    # Scale ventilation action level to a range of 0 to 1
    intensity = action_level_ventilation / 10
    
    # If ventilation is on, decrease moisture towards min limit
    if intensity > 0:
        new_modified_moisture = previous_modified_moisture - (previous_modified_moisture - min_limit) * intensity * action_factor
    else:
        # Decay moisture towards base moisture when no action is applied
        new_modified_moisture = previous_modified_moisture + (current_moisture_base - previous_modified_moisture) * decay_rate
    
    return new_modified_moisture