import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import uniform_filter1d

from Process_Data import *
from Signal_Models import *
from Decision_Making import *
from Plot import *

# Configuration Parameters - Easily Tunable
# Temperature Parameters
base_temp = 15
temperature_amplitude = 5
temperature_noise_level = 3
temperature_min_limit = 5
temperature_max_limit = 25
temperature_decay_rate = 0.05
temperature_action_factor = 0.05

# Humidity Parameters
base_humidity = 70
humidity_rain_impact = 0.5
humidity_drying_speed = 0.2
humidity_amplitude = 10
humidity_noise_level = 6
humidity_spike_chance = 0.05
humidity_spike_value = 15
humidity_min_limit = 0
humidity_decay_rate = 0.05
humidity_action_factor = 0.03

# Moisture Parameters
base_moisture = 25
moisture_rain_impact = 0.5
moisture_drying_speed = 0.2
moisture_noise_level = 3
moisture_min_limit = 0
moisture_decay_rate = 0.05
moisture_action_factor = 0.05

# Rain Parameters
rain_probability = 0.2
rain_intensity = 1

def main():
    # Generate time data with a 24-hour duration and 2-minute resolution
    time = generate_time_vector(24, 2)

    # Generate rain data with specified probability and intensity
    rain_data = generate_rain_vector(time, rain_probability=rain_probability, intensity=rain_intensity)

    # Base functions to simulate temperature, humidity, and moisture with noise
    temperature_base, temperature_noise, _ = simulate_temperature(
        time, base_temp=base_temp, amplitude=temperature_amplitude, noise_level=temperature_noise_level)
    humidity_base, humidity_noise, _ = simulate_humidity(
        time, rain_data, base_humidity=base_humidity, amplitude=humidity_amplitude, rain_impact=humidity_rain_impact,
        drying_speed=humidity_drying_speed, noise_level=humidity_noise_level, spike_chance=humidity_spike_chance,
        spike_value=humidity_spike_value)
    moisture_base, moisture_noise, _ = simulate_moisture(
        time, rain_data, base_moisture=base_moisture, rain_impact=moisture_rain_impact, drying_speed=moisture_drying_speed,
        noise_level=moisture_noise_level)
    
    # Initialize updated data arrays with base data values
    temperature_updated = np.copy(temperature_base)
    humidity_updated = np.copy(humidity_base)
    moisture_updated = np.copy(moisture_base)

    # Initialize action level arrays for dehumidifier, ventilation, and heating
    dehumidifier_action = np.zeros_like(time)
    ventilation_action = np.zeros_like(time)
    heating_action = np.zeros_like(time)

    # Loop over each time point to update and apply actions
    for i in range(len(time)):
        time_stamp = i

        if time_stamp > 0:
            # Update temperature, humidity, and moisture with actions and decay
            temperature_updated[i] = update_temperature(
                time_stamp, temperature_base, temperature_updated, heating_action[i - 1],
                min_limit=temperature_min_limit, max_limit=temperature_max_limit,
                decay_rate=temperature_decay_rate, action_factor=temperature_action_factor)
            humidity_updated[i] = update_humidity(
                time_stamp, humidity_base, humidity_updated, dehumidifier_action[i - 1], ventilation_action[i - 1],
                min_limit=humidity_min_limit, decay_rate=humidity_decay_rate, action_factor=humidity_action_factor)
            moisture_updated[i] = update_moisture(
                time_stamp, moisture_base, moisture_updated, ventilation_action[i - 1],
                min_limit=moisture_min_limit, decay_rate=moisture_decay_rate, action_factor=moisture_action_factor)

        # Add noise to the updated data
        noisy_temperature = temperature_updated + temperature_noise
        noisy_humidity = humidity_updated + humidity_noise
        noisy_moisture = moisture_updated + moisture_noise

        # Smooth noisy data for temperature, humidity, and moisture
        temperature = smooth_data(time_stamp, noisy_temperature)
        humidity = smooth_data(time_stamp, noisy_humidity)
        moisture = smooth_data(time_stamp, noisy_moisture)

        # Sensor data array for controllers
        sensdata = [temperature, humidity, moisture]

        if i % 1 == 0:
            # Determine actions for dehumidifier, ventilation, and heating based on sensor data (allows to run actions at larger intervals)
            dehumidifier_action[i] = dehumidifier_controller(sensdata)
            ventilation_action[i] = ventilation_controller(sensdata)
            heating_action[i] = heating_controller(sensdata)
        else:
            # Maintain the previous action level
            dehumidifier_action[i] = dehumidifier_action[i - 1]
            ventilation_action[i] = ventilation_action[i - 1]
            heating_action[i] = heating_action[i - 1]

    # Organize action levels and labels for plotting
    action_levels = np.array([dehumidifier_action, ventilation_action, heating_action])
    action_labels = ['Dehumidifier Action', 'Ventilation Action', 'Heating Action']

    # Organize updated, base, and noise data for plotting
    updated_data = [temperature_updated, humidity_updated, moisture_updated]
    base_data = [temperature_base, humidity_base, moisture_base]
    noise_data = [temperature_noise + temperature_base, humidity_noise + humidity_base, moisture_noise + moisture_base]
    labels = ['Temperature', 'Humidity', 'Moisture']

    # Plot action levels and updated sensor data
    plot_actions(time, action_levels, action_labels)
    plot_updated(time, updated_data, base_data, noise_data, labels)

    # Calculate rise times and stability for temperature, humidity, and moisture
    rise_temperature = time[(temperature_updated <= 22) & (temperature_updated >= 18)]
    rise_humidity = time[(humidity_updated <= 60) & (humidity_updated >= 40)]
    rise_moisture = time[moisture_updated <= 20]

    # Stability calculations based on the rise time
    stability_temperature = len(rise_temperature) / len(time[time >= rise_temperature[0]])
    stability_humidity = len(rise_humidity) / len(time[time >= rise_humidity[0]])
    stability_moisture = len(rise_moisture) / len(time[time >= rise_moisture[0]])

    # Output rise time and stability information
    print(f"Rise time for temperature: {rise_temperature[0]}")
    print(f"Rise time for humidity: {rise_humidity[0]}")
    print(f"Rise time for moisture: {rise_moisture[0]}")

    print(f"Stability for temperature: {stability_temperature}")
    print(f"Stability for humidity: {stability_humidity}")
    print(f"Stability for moisture: {stability_moisture}")

if __name__ == '__main__':
    sys.exit(main())