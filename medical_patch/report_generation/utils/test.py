# import numpy as np
# from scipy import signal
# import neurokit2 as nk
# import motor.motor_asyncio
# import biosppy.signals.ecg as ecg

# import asyncio
# from bson import ObjectId
# from datetime import datetime, timezone
# import time

# # MongoDB connection setup
# connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
# client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
# db = client['medicalacculive']
# collection = db['ecgdatas']

# # Ensure indexes on userId and date_time for faster query performance
# async def create_indexes():
#     await collection.create_index([('userId', 1), ('date_time', 1)])

# # Function to fetch data for a given time range and user ID
# async def fetch_data_chunk(userId, startTime, endTime):
#     start_date = datetime.fromtimestamp(startTime / 1000, tz=timezone.utc)
#     end_date = datetime.fromtimestamp(endTime / 1000, tz=timezone.utc)
#     pipeline = [
#         {
#             '$match': {
#                 'userId': ObjectId(userId),
#                 'date_time': {
#                     '$gte': start_date,
#                     '$lte': end_date,
#                 },
#             },
#         },
#         {
#             '$project': {
#                 '_id': 0,
#                 'date_time': 1,
#                 'ecg_vals': 1,
#             },
#         },
#     ]

#     try:
#         cursor = collection.aggregate(pipeline)
#         result = await cursor.to_list(length=None)
#         return result
#     except Exception as e:
#         print(f"Error fetching data: {e}")
#         return []  # Return empty list on error

# # Process ECG data chunk
# def process_ecg_data_chunk(data_chunk, sample_rate):
#     ecg_signal = data_chunk.get('ecg_vals', [])
    
#     # Example processing steps (filters and peaks extraction)
#     processed_data = process_ecg(ecg_signal, sample_rate)
#     processed_data["date_time"] = data_chunk['date_time']
#     return processed_data

# # Example ECG processing (replace with actual processing logic)
# def process_ecg(ecg_signal, sample_rate):
#     # High-pass filter
#     cutoff_freq = 0.5
#     b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
#     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)

#     # Band-pass filter
#     low_cutoff = 0.5
#     high_cutoff = 50
#     b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
#     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)

#     # Normalization
#     ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)
    
#     # Processing ECG
#     try:
#         out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
#         r_peaks = out['rpeaks']
#     except Exception as e:
#         print("Error in ECG processing:", e)
#         r_peaks = []
    
#     _, waves_peak = nk.ecg_delineate(ecg_signal_normalized, r_peaks, sampling_rate=sample_rate, method="peak")
    
#     # Handle NaN values in peaks
#     def handle_nan(peaks):
#         return [0 if np.isnan(x) else x for x in peaks[:-1]]

#     p_peaks = handle_nan(waves_peak['ECG_P_Peaks'])
#     q_peaks = handle_nan(waves_peak['ECG_Q_Peaks'])
#     s_peaks = handle_nan(waves_peak['ECG_S_Peaks'])
#     t_peaks = handle_nan(waves_peak['ECG_T_Peaks'])

#     ecg_signal_processed_list = ecg_signal_normalized.tolist()
#     processed_data = {
#         "ecg_vals": ecg_signal_processed_list,
#         "r_peaks": r_peaks,
#         "p_peaks": p_peaks,
#         "q_peaks": q_peaks,
#         "s_peaks": s_peaks,
#         "t_peaks": t_peaks
#     }
#     return processed_data

# async def fetch_and_process_data(chunk):
#     userId, startTime, endTime = chunk
#     result = await fetch_data_chunk(userId, startTime, endTime)
    
#     if result:
#         sample_rate = 200  # Replace with actual sampling rate
#         processed_chunks = [process_ecg_data_chunk(data_chunk, sample_rate) for data_chunk in result]
#         return processed_chunks
#     return []

# async def fetch_data_async(userId, startTime, endTime):
#     chunk_size = 900 * 1000  # Adjust chunk size as needed
#     chunks = []
#     current_time = int(startTime)
#     endTime = int(endTime)
    
#     while current_time < endTime:
#         next_time = min(current_time + chunk_size, endTime)
#         chunks.append((userId, current_time, next_time))
#         current_time = next_time
    
#     # Create tasks for each chunk
#     tasks = [fetch_and_process_data(chunk) for chunk in chunks]
#     results = await asyncio.gather(*tasks)
    
#     # Flatten the list of results
#     flattened_results = [item for sublist in results for item in sublist]
#     return flattened_results

# # Example usage66277eaf7dcc5669805e807e&startDate=1713830400000&endDate=1713916800000
# async def main():
#     user_id = "66277eaf7dcc5669805e807e"  # Replace with actual user ID
#     start_time = 1713830400000  # Replace with actual start time in milliseconds
#     end_time = 1713916800000  # Replace with actual end time in milliseconds
    
#     await create_indexes()
    
#     start = time.time()
#     data = await fetch_data_async(user_id, start_time, end_time)
#     end = time.time()
    
#     if data:
#         print(f"Fetched {len(data)} records in {end - start:.2f} seconds")
#     else:
#         print("No records fetched")

# if __name__ == "__main__":
#     asyncio.run(main())
import numpy as np
from scipy import signal
import neurokit2 as nk
import biosppy.signals.ecg as ecg
import motor.motor_asyncio
import asyncio
from bson import ObjectId
from datetime import datetime, timezone
import csv
import time

# MongoDB connection setup
connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
db = client['medicalacculive']
collection = db['ecgdatas']

# Ensure indexes on userId and date_time for faster query performance
async def create_indexes():
    await collection.create_index([('userId', 1), ('date_time', 1)])

# Function to fetch data for a given time range and user ID
async def fetch_data_chunk(userId, startTime, endTime):
    start_date = datetime.fromtimestamp(startTime / 1000, tz=timezone.utc)
    end_date = datetime.fromtimestamp(endTime / 1000, tz=timezone.utc)
    pipeline = [
        {
            '$match': {
                'userId': ObjectId(userId),
                'date_time': {
                    '$gte': start_date,
                    '$lte': end_date,
                },
            },
        },
        {
            '$project': {
                '_id': 0,
                'createdAt': 0,
                'updatedAt': 0,
                'userId': 0,
                'mac_address_framed': 0,
            },
        },
    ]

    try:
        cursor = collection.aggregate(pipeline)
        result = await cursor.to_list(length=None)
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []  # Return empty list on error

def process_ecg(ecg_signal, sample_rate):
    # High-pass filter
    cutoff_freq = 0.5
    b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)

    # Band-pass filter
    low_cutoff = 0.5
    high_cutoff = 50
    b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)

    # Normalization
    ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)
    
    # Processing ECG
    out = None
    try:
        out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
        r_peaks = out['rpeaks']
        if len(r_peaks) < 2:
            raise ValueError("Not enough beats to compute heart rate.")
    except Exception as e:
        print("Error:", e)
        r_peaks = []
    
    r_peaks = out['rpeaks']
    _, waves_peak = nk.ecg_delineate(ecg_signal_normalized, r_peaks, sampling_rate=sample_rate, method="peak")
    
    p_peaks = [0 if np.isnan(x) else x for x in waves_peak['ECG_P_Peaks'][:-1]]
    q_peaks = [0 if np.isnan(x) else x for x in waves_peak['ECG_Q_Peaks'][:-1]]
    s_peaks = [0 if np.isnan(x) else x for x in waves_peak['ECG_S_Peaks'][:-1]]
    t_peaks = [0 if np.isnan(x) else x for x in waves_peak['ECG_T_Peaks'][:-1]]

    ecg_signal_processed_list = ecg_signal_normalized.tolist()
    processed_data = {
        "ecg_vals": ecg_signal_processed_list,
        "out": r_peaks.tolist(),
        "r_peaks": r_peaks,
        "p_peaks": p_peaks,
        "q_peaks": q_peaks,
        "s_peaks": s_peaks,
        "t_peaks": t_peaks
    }
    return processed_data

async def fetch_and_process_data(chunk):
    userId, startTime, endTime = chunk
    result = await fetch_data_chunk(userId, startTime, endTime)
    if result:
        sample_rate = 200  # Example: replace with actual sampling rate
        processed_chunks = []

        for data_chunk in result:
            ecg_signal = data_chunk.get('ecg_vals', [])
            processed_data = process_ecg(ecg_signal, sample_rate)
            processed_data["date_time"] = data_chunk['date_time']
            processed_chunks.append(processed_data)
        return processed_chunks
    return []

async def fetch_data_async(userId, startTime, endTime):
    chunk_size = 900 * 1000  # Adjust chunk size as needed
    chunks = []
    current_time = int(startTime)
    endTime = int(endTime)
    while current_time < endTime:
        next_time = min(current_time + chunk_size, endTime)
        chunks.append((userId, current_time, next_time))
        current_time = next_time

    tasks = [fetch_and_process_data(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    fetched_data = [item for sublist in results if not isinstance(sublist, Exception) for item in sublist]
    
    return fetched_data

# Example usage
async def main():
    user_id = "66277eaf7dcc5669805e807e"  # Replace with actual user ID
    start_time = 1713830400000  # Replace with actual start time in milliseconds
    end_time = 1713916800000  # Replace with actual end time in milliseconds

    await create_indexes()

    start = time.time()
    data = await fetch_data_async(user_id, start_time, end_time)
    end = time.time()


    if data:
        print(f"Fetched {len(data)} records in {end - start} seconds")
    else:
        print("No records fetched")

if __name__ == "__main__":
    asyncio.run(main())

