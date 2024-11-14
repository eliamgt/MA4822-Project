import numpy as np

def simulate_temperature(time, base_temp, amplitude, noise_level):
    """
    Simulates temperature over time with a daily fluctuation and noise.

    Parameters:
    - time: Array of time points (in hours).
    - base_temp: The base temperature around which daily fluctuations occur.
    - amplitude: The amplitude of temperature fluctuations.
    - noise_level: The standard deviation of noise added to simulate variability.

    Returns:
    - temperature_base: The base temperature with daily fluctuation (sinusoidal).
    - noise: Noise values added to simulate randomness.
    - temperature_noisy: The simulated temperature with added noise.
    """

    # Generate the base temperature with daily sinusoidal fluctuation
    temperature_base = base_temp + amplitude * np.sin(2 * np.pi * (time / 24))
    # Generate random noise for each time point
    noise = np.random.normal(0, noise_level, size=temperature_base.shape)
    # Add noise to the base temperature to get the final noisy temperature
    temperature_noisy = temperature_base + noise
    
    return temperature_base, noise, temperature_noisy

def simulate_humidity(time, rain, base_humidity, amplitude, rain_impact, drying_speed, noise_level, spike_chance, spike_value):
    """
    Simulates humidity over time with a cumulative effect during rain, spikes, and noise.

    Parameters:
    - time: Array of time points (in hours).
    - rain: Array indicating rain occurrence (1 if raining, 0 if not).
    - base_humidity: The baseline humidity level.
    - amplitude: The amplitude of daily humidity fluctuations.
    - rain_impact: Incremental humidity increase during rain events.
    - drying_speed: Rate of humidity reduction when rain stops.
    - noise_level: The standard deviation of noise added to simulate variability.
    - spike_chance: Probability of random humidity spikes.
    - spike_value: Value of humidity increase during random spikes.

    Returns:
    - humidity_base: The base humidity with daily fluctuation (sinusoidal).
    - noise: Noise values including spikes and random noise.
    - humidity_noisy: The simulated humidity with cumulative rain effect and noise.
    """

    # Generate the base humidity with daily sinusoidal fluctuation
    humidity_base = base_humidity + amplitude * np.sin(2 * np.pi * (time / 24))
    # Initialize cumulative humidity based on base humidity
    cumulative_humidity = np.copy(humidity_base)
    
    # Apply cumulative effects due to rain
    for i in range(1, len(time)):
        if rain[i]:
            # Increase humidity during rain
            cumulative_humidity[i] = cumulative_humidity[i-1] + rain_impact
        else:
            # Gradually reduce humidity when it's not raining
            cumulative_humidity[i] = max(humidity_base[i], cumulative_humidity[i-1] - drying_speed)

    # Generate random spikes in humidity
    spikes = np.random.choice([0, spike_value], size=humidity_base.shape, p=[1 - spike_chance, spike_chance])
    # Generate noise and add it to the spikes
    noise = spikes + np.random.normal(0, noise_level, size=humidity_base.shape)
    # Add noise to the cumulative humidity to get the final noisy humidity
    humidity_noisy = cumulative_humidity + noise
    
    return humidity_base, noise, humidity_noisy

def simulate_moisture(time, rain, base_moisture, rain_impact, drying_speed, noise_level):
    """
    Simulates moisture accumulation over time, especially during rain events.

    Parameters:
    - time: Array of time points (in hours).
    - rain: Array indicating rain occurrence (1 if raining, 0 if not).
    - base_moisture: The baseline moisture level.
    - rain_impact: Incremental moisture increase during rain events.
    - drying_speed: Rate of moisture reduction when rain stops.
    - noise_level: The standard deviation of noise added to simulate variability.

    Returns:
    - moisture_base: The baseline moisture level.
    - noise: Noise values added to simulate randomness.
    - moisture_noisy: The simulated moisture with cumulative rain effect and noise.
    """

    # Initialize the base moisture level for each time point
    moisture_base = np.full_like(time, base_moisture, dtype=float)
    # Initialize cumulative moisture based on base moisture
    cumulative_moisture = np.copy(moisture_base)
    
    # Apply cumulative effects due to rain
    for i in range(1, len(time)):
        if rain[i]:
            # Accumulate moisture during rain
            cumulative_moisture[i] = cumulative_moisture[i-1] + rain_impact
        else:
            # Gradually reduce moisture when it's not raining
            cumulative_moisture[i] = max(moisture_base[i], cumulative_moisture[i-1] - drying_speed)

    # Generate noise for each time point
    noise = np.random.normal(0, noise_level, size=moisture_base.shape)
    # Add noise to the cumulative moisture to get the final noisy moisture
    moisture_noisy = cumulative_moisture + noise
    
    return moisture_base, noise, moisture_noisy

def generate_time_vector(duration, resolution):
    """
    Generates a time vector for a given duration and resolution.

    Parameters:
    - duration: Duration of the time vector in hours.
    - resolution: Time resolution in minutes.

    Returns:
    - time: Array of time points in hours.
    """

    # Generate time points from 0 to duration with specified resolution
    return np.arange(0, duration + resolution / 61, resolution / 60)

def generate_rain_vector(time, rain_probability, intensity):
    """
    Generates a rain vector where rain occurs with a certain probability.

    Parameters:
    - time: Array of time points.
    - rain_probability: Probability of rain occurrence at each time step.
    - intensity: Rain intensity (can be binary or float for intensity variation).

    Returns:
    - rain_vector: Array representing rain occurrence/intensity at each time step.
    """

    # Initialize rain vector with zeros (no rain)
    rain_vector = np.zeros_like(time)
    is_raining = False

    # Determine rain occurrence at each time step based on probability
    for i in range(len(time)):
        if is_raining:
            # Continue raining with specified intensity
            rain_vector[i] = intensity
            # Stop rain based on probability
            if np.random.rand() > rain_probability:
                is_raining = False
        else:
            # Start rain based on probability
            if np.random.rand() < rain_probability:
                is_raining = True
                rain_vector[i] = intensity
                
    return rain_vector