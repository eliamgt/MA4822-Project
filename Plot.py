import numpy as np
import matplotlib.pyplot as plt

def plot_actions(time, action_levels, labels=None):
    """
    Plots action levels over time and saves each plot as a PDF.

    Parameters:
    - time: Array of time points.
    - action_levels: List of action level arrays, where each array corresponds to a different action.
    - labels: List of labels for each action level plot (optional).

    This function creates a plot for each action level, sets the title if labels are provided,
    and saves each plot as a PDF. If labels are provided, the plot is saved with the label name.
    Otherwise, it's saved as 'action_plot_<index>.pdf'.
    """
    
    # Determine the number of actions to plot
    num_actions = len(action_levels)
    
    # Loop through each action level and create a separate plot
    for i in range(num_actions):
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot the action level over time
        ax.plot(time, action_levels[i])
        ax.set_ylabel('Action Level')
        
        # Set plot title if labels are provided
        if labels:
            ax.set_title(labels[i])
        
        # Add grid lines and set x-axis label
        ax.grid(True)
        ax.set_xlim(time[0], time[-1])
        plt.xlabel('Time (hours)')
        
        # Save plot as PDF with the label name or a default filename
        if labels:
            plt.savefig(f"{labels[i]}.pdf")
        else:
            plt.savefig(f"action_plot_{i}.pdf")
        
        # Close the figure to free up memory
        plt.close(fig)

def plot_updated(time, updated_data, base_data, noise_data, labels=None):
    """
    Plots updated, base, and noise data over time for each variable and saves each plot as a PDF.

    Parameters:
    - time: Array of time points.
    - updated_data: List of updated data arrays (one for each variable).
    - base_data: List of base data arrays (one for each variable).
    - noise_data: List of noise data arrays (one for each variable).
    - labels: List of labels for each variable plot (optional).

    This function creates a plot for each variable with base, updated, and noise data.
    It also adds a comfort range as horizontal lines for specific variables if labels are provided.
    Each plot is saved as a PDF with the label name or as 'plot_<index>.pdf' if no labels are provided.
    """
    
    # Determine the number of data series to plot
    num_data = len(updated_data)
    
    # Loop through each data series and create a separate plot
    for i in range(num_data):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot base, updated, and noise data for each variable
        ax.plot(time, base_data[i], label='Base Data')
        ax.plot(time, updated_data[i], label='Updated Data')
        ax.plot(time, noise_data[i], label='Noise Data')
        ax.set_ylabel('Value')
        
        # Set plot title and add comfort range lines if labels are provided
        if labels:
            ax.set_title(labels[i])
            # Add comfort range for specific labels
            if labels[i] == 'Temperature':
                ax.hlines([18, 22], time[0], time[-1], colors='r', linestyles='dashed', label='Comfort Range')
                ax.set_ylabel('Temperature (°C)')
            elif labels[i] == 'Humidity':
                ax.hlines([40, 60], time[0], time[-1], colors='r', linestyles='dashed', label='Comfort Range')
                ax.set_ylabel('Humidity (%)')
            elif labels[i] == 'Moisture':
                ax.hlines([20], time[0], time[-1], colors='r', linestyles='dashed', label='Comfort Range')
                ax.set_ylabel('Moisture (%)')
        
        # Add grid, legend, and set x-axis label
        ax.grid(True)
        ax.set_xlim(time[0], time[-1])
        ax.legend()
        plt.xlabel('Time (hours)')
        
        # Save plot as PDF with the label name or a default filename
        if labels:
            plt.savefig(f"{labels[i]}.pdf")
        else:
            plt.savefig(f"plot_{i}.pdf")
        
        # Close the figure to free up memory
        plt.close(fig)


