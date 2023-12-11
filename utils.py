import itertools
import threading
import time

import numpy as np
import pyaudio
import soundfile as sf
from scipy.signal import butter, lfilter, find_peaks
from collections import deque


def bandpass(data: np.ndarray, low: float, high: float, fs: int, order: int = 2) -> np.ndarray:
    """
    Apply a bandpass filter to the input data.

    Args:
        data (np.ndarray): Input data.
        low (float): Low cutoff frequency in Hz.
        high (float): High cutoff frequency in Hz.
        fs (int): Sampling rate in Hz.
        order (int, optional): Filter order. Defaults to 2.

    Returns:
        np.ndarray: Filtered data.
    """
    b, a = butter(order, [low, high], fs=fs, btype='band', analog=False)
    y = lfilter(b, a, data)
    return y


def find_beat_peaks(peaks: list, sampling_rate: int, last_tic: list = []) -> np.ndarray:
    """
    Find the best set of three peaks representing beats.

    Args:
        peaks (np.ndarray): Detected peaks.
        prominences (np.ndarray): Peak prominences.
        sampling_rate (int): Sampling rate in Hz.
        last_tic (list, optional): Last detected tic. Defaults to [].

    Returns:
        np.ndarray: Best set of three peaks representing beats.
    """

    # Chronological order of shocks and their descriptions:
    # 1. Unlocking: the impulse jewel striking the notch
    # 2. Beginning of Impulse for the Escape Wheel: the escape wheel catches up with the impulse face of the pallet
    # 3. Beginning of the Balance Wheel Impulse: the notch catching up with the impulse jewel
    # 4. Drop: The escape wheel tooth strikes the locking face of the exit pallet and…
    # 5. Safety Action: …simultaneously, the lever hits the banking pin
    MAX_BEAT_DURATION = 20 * sampling_rate / 1000                           # ms * sampling_rate
    MIN_BEAT_DURATION = 5 * sampling_rate / 1000                            # ms * sampling_rate
    MAX_UNLOCKING_DURATION = 9 * sampling_rate / 1000                       # ms * sampling_rate
    MIN_UNLOCKING_DURATION = 2 * sampling_rate / 1000                       # ms * sampling_rate
    MAX_DROP_DURATION = 11 * sampling_rate / 1000                           # ms * sampling_rate
    MIN_DROP_DURATION = 3 * sampling_rate / 1000                            # ms * sampling_rate

    # Function to check if three peaks are close together
    def are_peaks_close(peak_set):
        peak_locations = [peak[0] for peak in peak_set]
        beat_duration = peak_locations[2] - peak_locations[0]
        unlocking_duration = peak_locations[1] - peak_locations[0]
        drop_duration = peak_locations[2] - peak_locations[1]

        if len(last_tic) > 0:
            if (any(((peak_locations - last_tic) < - (10 * sampling_rate / 1000))) or
                    any(((peak_locations - last_tic) > (10 * sampling_rate / 1000)))):
                        return None

        return (MIN_BEAT_DURATION <= beat_duration <= MAX_BEAT_DURATION and
                MIN_UNLOCKING_DURATION <= unlocking_duration <= MAX_UNLOCKING_DURATION and
                MIN_DROP_DURATION <= drop_duration <= MAX_DROP_DURATION)

    # Find all sets of three peaks that are close together
    close_peak_sets = [list(sorted(comb)) for comb in itertools.combinations(peaks, 3) if are_peaks_close(comb)]
    if len(close_peak_sets) < 1:
        return np.array([])

    # Calculate prominence sums using the height from the tuples
    prominence_sums = [sum(peak[1] for peak in peak_set) for peak_set in close_peak_sets]
    best_peak_set = close_peak_sets[np.argmax(prominence_sums)]
    return np.array([peak[0] for peak in best_peak_set])


def moving_std(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the moving standard deviation of the given data.

    Args:
        data (np.ndarray): Input data.
        window_size (int): Size of the moving window.

    Returns:
        np.ndarray: Moving standard deviation values.
    """
    stds = np.array([np.std(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])
    # Padding to make the length equal to the original data
    pad_size = window_size - 1
    return np.pad(stds, (pad_size - pad_size // 2, pad_size // 2), mode='constant', constant_values=0)


def moving_std_dev(data, window_size):
    if len(data) < window_size:
        # Handle case where data is shorter than the window size
        return None

    sum_ = sum(data[:window_size])
    sum_of_squares = sum(x ** 2 for x in data[:window_size])
    std_devs = []

    for i in range(len(data) - window_size + 1):
        mean = sum_ / window_size
        variance = (sum_of_squares / window_size) - (mean ** 2)
        std_dev = np.sqrt(variance)
        std_devs.append(std_dev)

        # Update sums for next window
        if i + window_size < len(data):
            sum_ -= data[i]
            sum_ += data[i + window_size]
            sum_of_squares -= data[i] ** 2
            sum_of_squares += data[i + window_size] ** 2

    # Padding to make the length equal to the original data
    pad_size = window_size - 1
    return np.pad(std_devs, (pad_size - pad_size // 2, pad_size // 2), mode='constant', constant_values=0)


def moving_std_dev_welford(data, window_size):
    if len(data) < window_size:
        return None

    def update(existingAggregate, newValue, oldValue=None):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        if oldValue is not None:
            count -= 1
            delta = oldValue - mean
            mean -= delta / count
            delta2 = oldValue - mean
            M2 -= delta * delta2

        return (count, mean, M2)

    def finalize(existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float('nan')
        else:
            variance = M2 / count
            return np.sqrt(variance)

    std_devs = []
    aggregate = (0, 0.0, 0.0)

    # Initialize aggregate for the first window
    for i in range(window_size):
        aggregate = update(aggregate, data[i])

    for i in range(len(data) - window_size + 1):
        std_dev = finalize(aggregate)
        std_devs.append(std_dev)

        if i + window_size < len(data):
            aggregate = update(aggregate, data[i + window_size], data[i])

    # Padding to make the length equal to the original data
    pad_size = window_size - 1
    return np.pad(std_devs, (pad_size - pad_size // 2, pad_size // 2), mode='constant', constant_values=0)


def guess_bph(signal_data: np.ndarray, fs: int) -> float:
    """
    Estimate beats per hour (BPH) from the input signal data.

    Args:
        signal_data (np.ndarray): Input signal data.
        fs (int): Sampling rate in Hz.

    Returns:
        float: Estimated BPH.
    """
    # Convert signal data to 1 second sample
    # signal_data = np.abs(np.array(signal_data[:fs]))

    # Smooth the data
    signal_data = moving_std_dev_welford(signal_data[:fs], int(1 * fs / 1000))


    # Correlate the data and compute the power spectrum
    correlation = np.correlate(signal_data, signal_data, mode='full')
    correlation = correlation[len(correlation) // 2:]
    peaks, properties = find_peaks(correlation, distance=(fs/10)*0.99, prominence=0)
    peaks = peaks[peaks >= (fs/10)*0.99]
    bps = round(fs / peaks[0])

    ### OLD CALCULATIONS USING POWR SPECTRAL DENSITY (LESS ROBUST THAN CURRENT METHOD) ###
    # power_spectral_density = np.abs(np.fft.fft(correlation)) ** 2
    # bps = np.argmax(power_spectral_density[:len(power_spectral_density) // 2][5:10]) + 5

    return bps * 3600


def read_audio(path: str) -> tuple:
    """
    Read audio data from a file.

    Args:
        path (str): The file path of the audio.

    Returns:
        tuple: A tuple containing audio data as a NumPy array and the sample rate (fs).
    """
    try:
        # Load the audio file using soundfile
        audio, fs = sf.read(path)

        # Check if the loaded audio is empty or contains NaN values
        if audio.size == 0 or np.isnan(audio).any():
            raise ValueError("Empty or invalid audio file")

        # Convert from stereo to mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        return audio, fs
    except Exception as e:
        print(f"Error reading audio file '{path}': {str(e)}")
        return None, None


def write_audio(audio, fs, path, format='mp3'):

    """
    Save audio data to a file.

    Parameters:
    - audio (numpy.ndarray): The audio data as a NumPy array.
    - fs (int): The sample rate of the audio.
    - path (str): The file path where the audio should be saved.
    - format (str, optional): The audio file format ('wav' or 'mp3'). Default is 'wav'.
    """
    try:
        # Ensure the format is lower case
        format = format.lower()

        # Check if the format is supported
        if format not in ['wav', 'mp3']:
            raise ValueError("Unsupported audio format. Supported formats are 'wav' and 'mp3'.")

        # Check if the audio data is valid
        if audio.size == 0 or any(np.isnan(audio)):
            raise ValueError("Invalid audio data")

        # Save the audio to the specified file format
        sf.write(path, audio, fs, format=format)

        print(f"Audio saved to '{path}' in {format} format successfully.")
    except Exception as e:
        print(f"Error saving audio to '{path}': {str(e)}")


def get_microphone_sampling_frequency() -> int:
    """
    Get the sampling frequency of the active microphone.

    Returns:
        int: The sampling frequency in Hz.

    Raises:
        Exception: If no active microphone is found.
    """
    p = pyaudio.PyAudio()

    try:
        # Get the number of available audio devices
        num_devices = p.get_device_count()

        # Find the active microphone (you can use device_index=None for the default input device)
        for i in range(num_devices):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0 and device_info['hostApi'] == p.get_host_api_info_by_index(0)[
                'index']:
                active_device_index = i
                break
        else:
            raise Exception("No active microphone found.")

        # Get the sampling frequency of the active microphone
        sampling_frequency = int(device_info['defaultSampleRate'])

        return sampling_frequency

    finally:
        p.terminate()


def extract_peaks(queue, fs, liftangle, labels_queue, tics_queue):
    """
    Extract peaks from audio data and calculate various measurements.

    Args:
        queue (queue.Queue): A queue containing audio data.
        fs (int): The sampling frequency of the audio data.
        liftangle (float): The lift angle of the timepiece.
        labels_queue (queue.Queue): A queue to store measurement labels.
        tics_queue (queue.Queue): A queue to store tics and tacs data.

    Note:
        This function continuously processes audio data from the input queue and extracts tics and tacs peaks.
        It calculates various measurements such as amplitude, day rate, and beat error and stores them in the labels_queue.
    """
    tics = deque(maxlen=100)
    tacs = deque(maxlen=100)
    chunk_count = 0

    while True:
        if not queue.empty():
            audio = queue.get()

            # Smooth the signal
            audio = bandpass(audio, 5000, 11000, fs)
            audio = audio / max(np.abs(audio))
            _moving_std = moving_std_dev_welford(audio, int(2 * fs / 1000))

            # Extract parameters
            chunk_size = len(audio)
            t = (chunk_count * chunk_size) / fs

            # Find peaks
            peaks, attrs = find_peaks(_moving_std, height=np.std(_moving_std), width=int(1 * fs / 1000))

            # Create tuples of (peak_location, peak_height)
            peak_tuples = [(peak, attrs["peak_heights"][i]) for i, peak in enumerate(peaks)]

            # Split peaks into tic and tac based on their locations
            tic_peak_tuples = [pt for pt in peak_tuples if pt[0] < (chunk_size / 2)]
            tac_peak_tuples = [pt for pt in peak_tuples if pt[0] >= (chunk_size / 2)]

            # Extract tics and tacs from the peak tuples
            tic = find_beat_peaks(tic_peak_tuples, fs)
            tac = find_beat_peaks(tac_peak_tuples, fs)

            # Add the tic and tac to their respective deque
            if len(tic) == len(tac) == 3:
                tics.append(tic)
                tacs.append(tac)
                tics_queue.put([t, tic, tac])

            # Calculate
            if not chunk_count % 10 and len(tics) > 10:
                # Convert to array
                temp_tics = np.array(tics)
                temp_tacs = np.array(tacs)

                # Compute the global mean and std of the tics and tacs
                tmp_mean_tics = np.mean(temp_tics)
                tmp_std_tics = np.std(temp_tics)
                tmp_mean_tacs = np.mean(temp_tacs)
                tmp_std_tacs = np.std(temp_tacs)

                # Define inlier range for each array
                std_multiplier = 1
                inlier_range_tics = (
                tmp_mean_tics - std_multiplier * tmp_std_tics, tmp_mean_tics + std_multiplier * tmp_std_tics)
                inlier_range_tacs = (
                tmp_mean_tacs - std_multiplier * tmp_std_tacs, tmp_mean_tacs + std_multiplier * tmp_std_tacs)

                # Identify inliers and outliers in each array
                inliers_tics = (np.mean(temp_tics, axis=1) >= inlier_range_tics[0]) & (
                            np.mean(temp_tics, axis=1) <= inlier_range_tics[1])
                inliers_tacs = (np.mean(temp_tacs, axis=1) >= inlier_range_tacs[0]) & (
                            np.mean(temp_tacs, axis=1) <= inlier_range_tacs[1])

                # Remove outliers
                temp_tics = temp_tics[inliers_tics & inliers_tacs]
                temp_tacs = temp_tacs[inliers_tics & inliers_tacs]

                # Calculate amplitude
                theta = 2 * np.pi * 3 * (np.mean([np.mean(temp_tics[:, 2] - temp_tics[:, 0]),
                                                  np.mean(temp_tacs[:, 2] - temp_tacs[:, 0])]) / fs)
                amplitude = round(liftangle / np.sin(theta), 1)

                # Calculate day rate and beat error
                tic_tacs = list(zip(np.mean(temp_tics, axis=1), np.mean(temp_tacs, axis=1)))
                tic_tacs = [item + tic_idx * chunk_size for tic_idx, pair in enumerate(tic_tacs) for item in pair]
                diffs = np.diff(tic_tacs)
                rate_seconds_per_hour = 3600 / np.mean(diffs)
                beat_error = (np.abs(np.mean(diffs[list(range(0, len(tic_tacs) - 1, 2))]) -
                                     np.mean(diffs[list(range(1, len(tic_tacs) - 1, 2))])) / fs)
                rate, beat_error = round(rate_seconds_per_hour * 24, 1), round(beat_error, 4)

                # Store features in a queue
                labels_queue.put([amplitude, rate, beat_error])

            chunk_count += 1

        else:
            time.sleep(0.01)
