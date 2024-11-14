import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import numpy as np

def dehumidifier_controller(sensdata):
    """
    Fuzzy logic controller to determine the dehumidifier action level based on humidity levels.
    
    Parameters:
    sensdata - Array of sensor readings [humidity]

    Returns:
    dehumidifier_action - Dehumidifier action level (0-10)
    """

    # Define fuzzy variables
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    dehumidifier = ctrl.Consequent(np.arange(-10, 11, 1), 'dehumidifier')

    # Define membership functions for humidity
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
    humidity['medium'] = fuzz.trimf(humidity.universe, [45, 60, 75])
    humidity['high'] = fuzz.trimf(humidity.universe, [70, 100, 100])

    # Define membership functions for dehumidifier
    dehumidifier['off'] = fuzz.trimf(dehumidifier.universe, [-10, -10, 1])
    dehumidifier['low'] = fuzz.trimf(dehumidifier.universe, [0, 1.5, 3])
    dehumidifier['high'] = fuzz.trimf(dehumidifier.universe, [2, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(humidity['high'], dehumidifier['high'])
    rule2 = ctrl.Rule(humidity['medium'], dehumidifier['low'])
    rule3 = ctrl.Rule(humidity['low'], dehumidifier['off'])

    # Create control system
    dehumidifier_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    dehumidifier_sim = ctrl.ControlSystemSimulation(dehumidifier_ctrl)

    dehumidifier_sim.input['humidity'] = sensdata[1]

    # Compute the output
    dehumidifier_sim.compute()

    # Get the output value
    dehumidifier_action = dehumidifier_sim.output['dehumidifier']

    return dehumidifier_action

def ventilation_controller(sensdata):
    """
    Fuzzy logic controller to determine the ventilation action level based on humidity and moisture levels.
    
    Parameters:
    sensdata - Array of sensor readings [humidity, moisture]

    Returns:
    ventilation_action - Ventilation action level (0-10)
    """

    # Define fuzzy variables
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
    ventilation = ctrl.Consequent(np.arange(-10, 11, 1), 'ventilation')

    # Define membership functions for humidity
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 60])
    humidity['medium'] = fuzz.trimf(humidity.universe, [55, 65, 75])
    humidity['high'] = fuzz.trimf(humidity.universe, [70, 100, 100])

    # Define membership functions for moisture
    moisture['low'] = fuzz.trimf(moisture.universe, [0, 0, 15])
    moisture['medium'] = fuzz.trimf(moisture.universe, [10, 15, 20])
    moisture['high'] = fuzz.trimf(moisture.universe, [15, 100, 100])

    # Define membership functions for ventilation
    ventilation['off'] = fuzz.trimf(ventilation.universe, [-10, -10, 1])
    ventilation['low'] = fuzz.trimf(ventilation.universe, [0, 1.5, 3])
    ventilation['high'] = fuzz.trimf(ventilation.universe, [2, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(humidity['high'] & moisture['high'], ventilation['high'])
    rule2 = ctrl.Rule(humidity['high'] & moisture['medium'], ventilation['high'])
    rule3 = ctrl.Rule(humidity['high'] & moisture['low'], ventilation['high'])
    rule4 = ctrl.Rule(humidity['medium'] & moisture['high'], ventilation['high'])
    rule6 = ctrl.Rule(humidity['low'] & moisture['high'], ventilation['high'])
    rule5 = ctrl.Rule(humidity['medium'] & moisture['medium'], ventilation['low'])
    rule7 = ctrl.Rule(humidity['medium'] & moisture['low'], ventilation['low'])
    rule8 = ctrl.Rule(humidity['low'] & moisture['medium'], ventilation['low'])
    rule9 = ctrl.Rule(humidity['low'] & moisture['low'], ventilation['off'])

    # Create control system
    ventilation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    ventilation_sim = ctrl.ControlSystemSimulation(ventilation_ctrl)

    ventilation_sim.input['humidity'] = sensdata[1]
    ventilation_sim.input['moisture'] = sensdata[2]

    # Compute the output
    ventilation_sim.compute()

    # Get the output value
    ventilation_action = ventilation_sim.output['ventilation']
    return ventilation_action

def heating_controller(sensdata):
    """
    Fuzzy logic controller to determine the heating action level based on temperature levels.
    
    Parameters:
    sensdata - Array of sensor readings [temperature, humidity]
    
    Returns:
    heating_action - Heating action level (-10-10)
    """

    # Define fuzzy variables
    temperature = ctrl.Antecedent(np.arange(0, 31, 1), 'temperature')
    heating = ctrl.Consequent(np.arange(-10, 11, 1), 'heating')

    # Define membership functions for temperature
    temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 19])
    temperature['medium'] = fuzz.trimf(temperature.universe, [18, 20, 22])
    temperature['hot'] = fuzz.trimf(temperature.universe, [21, 30, 30])

    # Define membership functions for heating
    heating['cool'] = fuzz.trimf(heating.universe, [-10, -10, 0])
    heating['off'] = fuzz.trimf(heating.universe, [-5, 0, 5])
    heating['on'] = fuzz.trimf(heating.universe, [0, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(temperature['hot'], heating['cool'])
    rule2 = ctrl.Rule(temperature['medium'], heating['off'])
    rule3 = ctrl.Rule(temperature['cold'], heating['on'])

    # Create control system
    heating_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    heating_sim = ctrl.ControlSystemSimulation(heating_ctrl)

    heating_sim.input['temperature'] = sensdata[0]

    # Compute the output
    heating_sim.compute()

    # Get the output value
    heating_action = heating_sim.output['heating']
    return heating_action

def plot_mf():
    # Define fuzzy variables
    temperature = ctrl.Antecedent(np.arange(0, 31, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
    heating = ctrl.Consequent(np.arange(-10, 11, 1), 'heating')
    dehumidifier = ctrl.Consequent(np.arange(-10, 11, 1), 'dehumidifier')
    ventilation = ctrl.Consequent(np.arange(-10, 11, 1), 'ventilation')

    # Define membership functions for temperature
    temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 21])
    temperature['medium'] = fuzz.trimf(temperature.universe, [20, 22.5, 25])
    temperature['hot'] = fuzz.trimf(temperature.universe, [24, 30, 30])

    # Define membership functions for humidity
    humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 60])
    humidity['medium'] = fuzz.trimf(humidity.universe, [55, 65, 75])
    humidity['high'] = fuzz.trimf(humidity.universe, [70, 100, 100])

    # Define membership functions for moisture
    moisture['low'] = fuzz.trimf(moisture.universe, [0, 0, 15])
    moisture['medium'] = fuzz.trimf(moisture.universe, [10, 15, 20])
    moisture['high'] = fuzz.trimf(moisture.universe, [15, 100, 100])

    # Define membership functions for heating
    heating['cool'] = fuzz.trimf(heating.universe, [-10, -10, 0])
    heating['off'] = fuzz.trimf(heating.universe, [-5, 0, 5])
    heating['on'] = fuzz.trimf(heating.universe, [0, 10, 10])

    # Define membership functions for dehumidifier
    dehumidifier['off'] = fuzz.trimf(dehumidifier.universe, [-10, -10, 1])
    dehumidifier['low'] = fuzz.trimf(dehumidifier.universe, [0, 1.5, 3])
    dehumidifier['high'] = fuzz.trimf(dehumidifier.universe, [2, 10, 10])

    # Define membership functions for ventilation
    ventilation['off'] = fuzz.trimf(ventilation.universe, [-10, -10, 1])
    ventilation['low'] = fuzz.trimf(ventilation.universe, [0, 1.5, 3])
    ventilation['high'] = fuzz.trimf(ventilation.universe, [2, 10, 10])

    # Plot the membership functions for temperature
    plt.plot(temperature.universe, temperature['cold'].mf, label='Cold')
    plt.plot(temperature.universe, temperature['medium'].mf, label='Medium')
    plt.plot(temperature.universe, temperature['hot'].mf, label='Hot')
    plt.title('Temperature Membership Functions')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_temperature.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot the membership functions for humidity
    plt.plot(humidity.universe, humidity['low'].mf, label='Low')
    plt.plot(humidity.universe, humidity['medium'].mf, label='Medium')
    plt.plot(humidity.universe, humidity['high'].mf, label='High')
    plt.title('Humidity Membership Functions')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_humidity.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot the membership functions for moisture
    plt.plot(moisture.universe, moisture['low'].mf, label='Low')
    plt.plot(moisture.universe, moisture['medium'].mf, label='Medium')
    plt.plot(moisture.universe, moisture['high'].mf, label='High')
    plt.title('Moisture Membership Functions')
    plt.xlabel('Moisture Level')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_moisture.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot the membership functions for heating action
    plt.plot(heating.universe, heating['cool'].mf, label='Cool')
    plt.plot(heating.universe, heating['off'].mf, label='Off')
    plt.plot(heating.universe, heating['on'].mf, label='On')
    plt.title('Heating Action Membership Functions')
    plt.xlabel('Heating Action Level')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_heating.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot the membership functions for dehumidifier action
    plt.plot(dehumidifier.universe, dehumidifier['off'].mf, label='Off')
    plt.plot(dehumidifier.universe, dehumidifier['low'].mf, label='Low')
    plt.plot(dehumidifier.universe, dehumidifier['high'].mf, label='High')
    plt.title('Dehumidifier Action Membership Functions')
    plt.xlabel('Dehumidifier Action Level')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_dehumidifier.pdf', format='pdf', dpi=300)
    plt.show()

    # Plot the membership functions for ventilation action
    plt.plot(ventilation.universe, ventilation['off'].mf, label='Off')
    plt.plot(ventilation.universe, ventilation['low'].mf, label='Low')
    plt.plot(ventilation.universe, ventilation['high'].mf, label='High')
    plt.title('Ventilation Action Membership Functions')
    plt.xlabel('Ventilation Action Level')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.savefig('mf_ventilation.pdf', format='pdf', dpi=300)
    plt.show()
