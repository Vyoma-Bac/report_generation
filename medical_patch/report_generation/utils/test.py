import asyncio
import aiohttp
from aiohttp import ClientSession
import numpy as np
from scipy import signal
import neurokit2 as nk
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime, timedelta
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from io import BytesIO
import os
import requests
import time
import csv
import time
from functools import wraps # Assuming you have installed 'ecg' package (e.g., pip install ecg)
from utils.utils import timer_decorator
import asyncio
import aiohttp
from aiohttp import ClientSession
import numpy as np
from scipy import signal
@timer_decorator
async def fetch_data(session, api_endpoint, userId, startTime, endTime):
    params = {
        'userId': userId,
        'startDate': startTime,
        'endDate': endTime
    }
    try:
        async with session.get(api_endpoint, params=params) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses
            return await response.json()
    except aiohttp.ClientError as e:
        print(f"Error fetching data from the API: {e}")
        return None
@timer_decorator
async def fetch_ecg_data_parallel(userId, startTime, endTime):
    api_endpoint = "https://api.accu.live/api/v1/devices/getecgdata"
    tasks = []
    
    async with ClientSession() as session:
        # Divide the time range into 2-hour chunks
        current_time = startTime
        while current_time < endTime:
            next_time = min(current_time + 7200, endTime)  # End of current 2-hour chunk
            tasks.append(fetch_data(session, api_endpoint, userId, current_time, next_time))
            current_time = next_time
        
        # Fetch all data concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process each chunk of ECG data
        ecg_vals = []
        for result in results:
            if result and result.get('data') and result['data']:  # Check if 'data' exists and is not empty
                for data_chunk in result['data']:
                    ecg_signal = data_chunk.get('ecg_vals', [])  # Get 'ecg_vals' from data chunk
                    sample_rate = 200  # Example: replace with actual sampling rate
                    
                    # Process ECG signal using process_ecg function
                    ecg_signal_processed, out = process_ecg(ecg_signal, sample_rate)
                    
                    # Convert processed ECG signal to list
                    ecg_signal_processed_list = ecg_signal_processed.tolist()
                    
                    # Ensure 'out' is JSON-serializable
                    out_serializable = {
                        "ts": out.ts.tolist() if isinstance(out.ts, np.ndarray) else out.ts,
                        "filtered": out.filtered.tolist() if isinstance(out.filtered, np.ndarray) else out.filtered,
                        "rpeaks": out.rpeaks.tolist() if isinstance(out.rpeaks, np.ndarray) else out.rpeaks,
                        "templates_ts": out.templates_ts.tolist() if isinstance(out.templates_ts, np.ndarray) else out.templates_ts,
                        # Add other fields as needed
                    }
                    # Prepare processed data structure with 'date_time' and processed 'ecg_vals'
                    processed_data = {
                        "date_time": data_chunk['date_time'],
                        "ecg_vals": ecg_signal_processed_list,
                        "out": out
                    }
                    
                    # Append processed_data to ecg_vals
                    ecg_vals.append(processed_data)
        
        return ecg_vals
@timer_decorator
def process_ecg(ecg_signal, sample_rate):
    # Remove baseline wander using a high-pass filter
    cutoff_freq = 0.5  # Cutoff frequency in Hz
    b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)
    
    # Apply bandpass filter to remove noise
    low_cutoff = 0.5  # Lower cutoff frequency in Hz
    high_cutoff = 50  # Higher cutoff frequency in Hz
    b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)
    
    # Normalize the ECG signal
    ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)
    
    # Initialize 'out' variable
    out = None
    
    # Process the ECG signal to detect R peaks
    try:
        out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
        r_peaks = out['rpeaks']
        if len(r_peaks) < 2:
            raise ValueError("Not enough beats to compute heart rate.")
    except Exception as e:
        print("Error:", e)
        r_peaks = []  # Return empty list if an error occurs
    
    return ecg_signal_normalized, out

# # Example usage:
# async def main():
#     userId = "your_user_id"
#     startTime = 1623888000  # Replace with your UTC start date in seconds
#     endTime = 1623974400    # Replace with your UTC end date in seconds

#     ecg_vals = await fetch_ecg_data_parallel(userId, startTime, endTime)

#     for i, processed_data in enumerate(ecg_vals):
#         print(f"Processed ECG Data {i+1}:")
#         print(processed_data)
#         print()

# if __name__ == '__main__':
#     asyncio.run(main())
