import numpy as np
from scipy import optimize, signal

def analyze_signal_for_labview(data_array, sample_rate):
    """
    Simplified signal analysis function for LabVIEW integration.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        1D array of signal values (amplitude)
    sample_rate : float
        Sampling rate in Hz
    
    Returns:
    --------
    dict
        Dictionary containing signal parameters
    """
    # Generate time array based on sample rate
    n_samples = len(data_array)
    time = np.arange(n_samples) / sample_rate
    
    # Stack time and data into a 2D array
    data = np.column_stack((time, data_array))
    
    # Call the main analysis function
    return analyze_signal(data, sample_rate)

def analyze_signal(data, sample_rate):
    """
    Analyzes a noisy signal to determine its parameters.
    """
    # Extract time and values
    if data.shape[1] == 2:
        time = data[:, 0]
        values = data[:, 1]
    else:
        # If only values are provided, generate time array
        values = data.flatten()
        time = np.arange(len(values)) / sample_rate
    
    # Calculate FFT to get preliminary frequency information
    n = len(values)
    fft_values = np.fft.rfft(values)
    fft_freqs = np.fft.rfftfreq(n, 1/sample_rate)
    
    # Find dominant frequency (excluding DC component)
    magnitude_spectrum = np.abs(fft_values)
    if len(magnitude_spectrum) > 1:
        dominant_idx = np.argmax(magnitude_spectrum[1:]) + 1
        dominant_freq = fft_freqs[dominant_idx]
    else:
        dominant_freq = 0
    
    # Estimate DC offset
    dc_offset = np.mean(values)
    
    # Detrend the signal to remove DC offset for easier analysis
    detrended_values = values - dc_offset
    
    # Estimate amplitude from standard deviation or peak-to-peak
    amplitude_estimate = np.std(detrended_values) * np.sqrt(2)
    
    # Detect signal type by analyzing harmonics and shape
    signal_type = detect_signal_type(fft_values, fft_freqs, dominant_freq, values)
    
    # Define different fitting functions based on signal type
    if signal_type == "sine":
        def fitting_function(t, amplitude, frequency, phase):
            return amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
    elif signal_type == "square":
        def fitting_function(t, amplitude, frequency, phase):
            return amplitude * signal.square(2 * np.pi * frequency * t + phase) + dc_offset
    elif signal_type == "triangular":
        def fitting_function(t, amplitude, frequency, phase):
            return amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase, 0.5) + dc_offset
    else:  # Default to sine if signal type is unknown
        signal_type = "sine"
        def fitting_function(t, amplitude, frequency, phase):
            return amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
    
    # Initial parameter estimate
    initial_params = [amplitude_estimate, dominant_freq, 0]
    bounds = ([0, 0, -np.pi], [np.inf, sample_rate/2, np.pi])
    
    # Optimize parameters using curve fitting
    try:
        optimal_params, _ = optimize.curve_fit(
            fitting_function, time, values, 
            p0=initial_params, 
            bounds=bounds,
            maxfev=5000
        )
        
        amplitude, frequency, phase = optimal_params
        
        # Compute fitted signal
        fitted_values = fitting_function(time, amplitude, frequency, phase)
        
        # Calculate SNR
        signal_power = np.var(fitted_values)
        noise = values - fitted_values
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
    except (RuntimeError, ValueError) as e:
        # If curve fitting fails, use the estimates
        amplitude = amplitude_estimate
        frequency = dominant_freq
        phase = 0
        snr_db = 0
    
    # Return parameters as a dictionary
    return {
        'signal_type': signal_type,
        'amplitude': float(amplitude),  # Convert to native Python float for better LabVIEW compatibility
        'frequency': float(frequency),
        'offset': float(dc_offset),
        'phase': float(phase),
        'snr': float(snr_db)
    }

def detect_signal_type(fft_values, fft_freqs, dominant_freq, time_domain_values):
    """
    Detects the type of signal based on harmonic content and waveform characteristics.
    """
    # Find peaks in FFT to analyze harmonic content
    # Normalize FFT values for easier analysis
    magnitude_spectrum = np.abs(fft_values)
    if np.max(magnitude_spectrum) > 0:
        normalized_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    else:
        return "unknown"
    
    # Find significant harmonics
    threshold = 0.1  # Threshold for significant harmonics
    harmonic_indices = np.where(normalized_spectrum > threshold)[0]
    
    # Get frequencies of significant harmonics
    harmonic_freqs = fft_freqs[harmonic_indices]
    
    # Remove DC component if present
    if len(harmonic_freqs) > 0 and harmonic_freqs[0] < 0.01 * dominant_freq:
        harmonic_freqs = harmonic_freqs[1:]
        harmonic_indices = harmonic_indices[1:]
    
    # No significant harmonics found
    if len(harmonic_freqs) == 0:
        return "unknown"
    
    # Only fundamental frequency is significant - likely a sine wave
    if len(harmonic_freqs) == 1:
        return "sine"
    
    # Analyze time domain for additional clues
    # Check crest factor (peak value / RMS value)
    peak_value = np.max(np.abs(time_domain_values - np.mean(time_domain_values)))
    rms_value = np.sqrt(np.mean(np.square(time_domain_values - np.mean(time_domain_values))))
    
    if rms_value > 0:
        crest_factor = peak_value / rms_value
        
        if 1.6 <= crest_factor <= 1.9:  # Close to sqrt(2), characteristic of sine waves
            return "sine"
        elif crest_factor >= 2.5:  # Square waves have higher crest factors
            return "square"
        elif 1.9 < crest_factor < 2.5:  # Triangular waves have intermediate crest factors
            return "triangular"
    
    return "unknown"

# Example of how the LabVIEW Python node would call our function
def labview_entry_point(signal_data, sample_rate):
    """
    Entry point for LabVIEW to call our function.
    
    Parameters:
    -----------
    signal_data : array-like
        1D array of signal data (amplitude values)
    sample_rate : float
        Sampling rate in Hz
    
    Returns:
    --------
    tuple
        signal_type, amplitude, frequency, offset, phase, snr
    """
    print("Signal Data:", signal_data)
    print("Sample Rate:", sample_rate)
    # Convert input data to numpy array if it's not already
    signal_array = np.array(signal_data)
    
    # Call our analysis function
    result = analyze_signal_for_labview(signal_array, sample_rate)
    
    # Return results as a tuple (easier for LabVIEW to handle)
    
    return [
        float(result['amplitude']), 
        float(result['frequency']), 
        float(result['offset']), 
        float(result['phase']), 
        float(result['snr'])
        ]

# signal_data = [0.1, 0.5, 0.3, -0.1, -0.5]
# sample_rate = 1000.0
# print(labview_entry_point(signal_data, sample_rate))