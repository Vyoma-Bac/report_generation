# import asyncio
# import aiohttp
# from aiohttp import ClientSession
# import numpy as np
# from scipy import signal
# import neurokit2 as nk
# import biosppy.signals.ecg as ecg
# import matplotlib.pyplot as plt
# from datetime import datetime
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen.canvas import Canvas
# from reportlab.lib.utils import ImageReader
# from reportlab.platypus import Table, TableStyle
# from reportlab.lib import colors
# from io import BytesIO
# from datetime import datetime, timedelta
# from reportlab.platypus import Paragraph
# from reportlab.lib.styles import ParagraphStyle
# from io import BytesIO
# import os
# import requests
# import time
# import csv
# import time
# from functools import wraps # Assuming you have installed 'ecg' package (e.g., pip install ecg)
# #from utils.utils import timer_decorator
# import asyncio
# import aiohttp
# from aiohttp import ClientSession
# import numpy as np
# from scipy import signal

# # # # # @timer_decorator
# # # # # async def fetch_data(session, api_endpoint, userId, startTime, endTime):
# # # # #     params = {
# # # # #         'userId': userId,
# # # # #         'startDate': startTime,
# # # # #         'endDate': endTime
# # # # #     }
# # # # #     print("fetchdataaaa")
# # # # #     try:
# # # # #         async with session.get(api_endpoint, params=params) as response:
# # # # #             response.raise_for_status()
# # # # #             return await response.json()
# # # # #     except aiohttp.ClientError as e:
# # # # #         print(f"Error fetching data from the API: {e}")
# # # # #         return None

# # # # # @timer_decorator
# # # # # def process_ecg(ecg_signal, sample_rate):
# # # # #     cutoff_freq = 0.5
# # # # #     b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
# # # # #     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)

# # # # #     low_cutoff = 0.5
# # # # #     high_cutoff = 50
# # # # #     b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
# # # # #     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)

# # # # #     ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)
# # # # #     out = None
# # # # #     try:
# # # # #         out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
# # # # #         r_peaks = out['rpeaks']
# # # # #         if len(r_peaks) < 2:
# # # # #             raise ValueError("Not enough beats to compute heart rate.")
# # # # #     except Exception as e:
# # # # #         print("Error:", e)
# # # # #         r_peaks = []
# # # # #     # r_peaks = out['rpeaks']
# # # # #     # _ ,waves_peak = nk.ecg_delineate(ecg_signal_normalized, r_peaks, sampling_rate=sample_rate, method="peak")
# # # # #     # p_peaks = waves_peak['ECG_P_Peaks']
# # # # #     # q_peaks = waves_peak['ECG_Q_Peaks']
# # # # #     # s_peaks = waves_peak['ECG_S_Peaks']
# # # # #     # t_peaks = waves_peak['ECG_T_Peaks']

# # # # #     # # filtering peaks
# # # # #     # p_peaks.pop()
# # # # #     # q_peaks.pop()
# # # # #     # s_peaks.pop()
# # # # #     # t_peaks.pop()
# # # # #     # p_peaks = [0 if np.isnan(x) else x for x in p_peaks]
# # # # #     # q_peaks = [0 if np.isnan(x) else x for x in q_peaks]
# # # # #     # s_peaks = [0 if np.isnan(x) else x for x in s_peaks]
# # # # #     # t_peaks = [0 if np.isnan(x) else x for x in t_peaks]

# # # # #     ecg_signal_processed_list = ecg_signal_normalized.tolist()
# # # # #     processed_data = {
# # # # #         "ecg_vals": ecg_signal_processed_list,
# # # # #         "out": r_peaks.tolist(),
# # # # #         # "r_peaks": r_peaks,
# # # # #         # "p_peaks" : p_peaks,
# # # # #         # "q_peaks" : q_peaks,
# # # # #         # "s_peaks" : s_peaks,
# # # # #         # "t_peaks" : t_peaks
# # # # #     }
# # # # #     print("hehehehe")
# # # # #     return processed_data

# # # # # @timer_decorator
# # # # # async def fetch_ecg_data_parallel(userId, startTime, endTime):
# # # # #     api_endpoint = "https://api.accu.live/api/v1/devices/getecgdata"
# # # # #     tasks = []
# # # # #     endTime = int(endTime)
# # # # #     async with ClientSession() as session:
# # # # #         current_time = int(startTime)
# # # # #         while current_time < endTime:
# # # # #             next_time = min(current_time + 3600, endTime)  # 2-hour chunks
# # # # #             tasks.append(fetch_and_process_data(session, api_endpoint, userId, current_time, next_time))
# # # # #             current_time = next_time

# # # # #         results = await asyncio.gather(*tasks, return_exceptions=True)
# # # # #         ecg_vals = [item for sublist in results if sublist for item in sublist]
# # # # #         print(ecg_vals)
# # # # #         return {"ecg_vals":"jhadj"}

# # # # # @timer_decorator
# # # # # async def fetch_and_process_data(session, api_endpoint, userId, startTime, endTime):
# # # # #     result = await fetch_data(session, api_endpoint, userId, startTime, endTime)
# # # # #     if result and result.get('data'):
# # # # #         sample_rate = 200  # Example: replace with actual sampling rate
# # # # #         processed_chunks = []
# # # # #         for data_chunk in result['data']:
# # # # #             ecg_signal = data_chunk.get('ecg_vals', [])
# # # # #             processed_data = process_ecg(ecg_signal, sample_rate)
# # # # #             processed_data["date_time"] = data_chunk['date_time']
# # # # #             processed_chunks.append(processed_data)
# # # # #         return processed_chunks
# # # # #     return None

# # # # # from pymongo import MongoClient
# # # # # from bson import ObjectId

# # # # # # Connection string
# # # # # connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"

# # # # # # Create a MongoClient
# # # # # client = MongoClient(connection_string)

# # # # # # Access the database
# # # # # db = client['medicalacculive']

# # # # # # Define the userId
# # # # # user_id = '66277eaf7dcc5669805e807e'  # Replace with the actual user ID

# # # # # # Define the aggregation pipeline
# # # # # pipeline = [
# # # # #     {
# # # # #         '$match': {
# # # # #             '_id': ObjectId(user_id),  # Match documents by _id
# # # # #         },
# # # # #     },
# # # # #     {
# # # # #         '$lookup': {
# # # # #             'from': 'doctercustomers',  # Join with doctercustomers collection
# # # # #             'localField': '_id',  # Match local _id with userId in doctercustomers
# # # # #             'foreignField': 'userId',
# # # # #             'as': 'doctor',  # Output results as doctor array
# # # # #         },
# # # # #     },
# # # # #     {
# # # # #         '$unwind': {
# # # # #             'path': '$doctor',  # Deconstruct doctor array
# # # # #             'preserveNullAndEmptyArrays': True,  # Keep documents without matches
# # # # #         },
# # # # #     },
# # # # #     {
# # # # #         '$lookup': {
# # # # #             'from': 'users',  # Join with users collection
# # # # #             'localField': 'doctor.docterId',  # Match doctor.docterId with _id in users
# # # # #             'foreignField': '_id',
# # # # #             'as': 'doctorDetail',  # Output results as doctorDetail array
# # # # #         },
# # # # #     },
# # # # #     {
# # # # #         '$unwind': {
# # # # #             'path': '$doctorDetail',  # Deconstruct doctorDetail array
# # # # #             'preserveNullAndEmptyArrays': True,  # Keep documents without matches
# # # # #         },
# # # # #     },
# # # # #     {
# # # # #         '$project': {  # Project desired fields
# # # # #             'first_Name': 1,
# # # # #             'last_Name': 1,
# # # # #             'DOB': 1,
# # # # #             'gender': 1,
# # # # #             'countryCode': 1,
# # # # #             'mobile_no': 1,
# # # # #             'imageUrl': 1,
# # # # #             'email': 1,
# # # # #             'nationality': 1,
# # # # #             'address': '$Address',
# # # # #             'height': 1,
# # # # #             'weight': 1,
# # # # #             'medical_history': 1,
# # # # #             'emergencyContacts': 1,
# # # # #             'doctorDetail': {  # Include fields from doctorDetail
# # # # #                 'first_Name': '$doctorDetail.first_Name',
# # # # #                 'last_Name': '$doctorDetail.last_Name',
# # # # #                 'DOB': '$doctorDetail.DOB',
# # # # #                 'email': '$doctorDetail.email',
# # # # #                 'gender': '$doctorDetail.gender',
# # # # #                 'countryCode': '$doctorDetail.countryCode',
# # # # #                 'mobile_no': '$doctorDetail.mobile_no',
# # # # #                 'address': '$doctorDetail.Address',
# # # # #                 'imageUrl': '$doctorDetail.imageUrl',
# # # # #                 'licenseDetail': '$doctorDetail.licenseDetail',
# # # # #             },
# # # # #         },
# # # # #     },
# # # # # ]

# # # # # # Run the aggregation pipeline
# # # # # result = list(db.users.aggregate(pipeline))

# # # # # # Print the results
# # # # # for doc in result:
# # # # #     print(doc)
# # # # from flask import jsonify
# # # # from pymongo import MongoClient
# # # # from bson import ObjectId
# # # # from datetime import datetime
# # # # def lol(userId, startTime, endTime):
# # # # # Connection string
# # # #     connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"

# # # #     # Create a MongoClient
# # # #     client = MongoClient(connection_string)

# # # #     # Access the database
# # # #     db = client['medicalacculive']

# # # #     # Define the userId and dates in UTC seconds
# # # #     # user_id = '66277eaf7dcc5669805e807e'  # Replace with the actual user ID
# # # #     # start_date_utc_milliseconds = 1713864900000  # Replace with the actual start date in UTC milliseconds
# # # #     # end_date_utc_milliseconds = 1713868200000    # Replace with the actual end date in UTC milliseconds

# # # #     # Convert UTC milliseconds to datetime objects
# # # #     start_date = datetime.fromtimestamp(startTime / 1000)
# # # #     end_date = datetime.fromtimestamp(endTime / 1000)

# # # #     print("Start Date:", start_date)
# # # #     print("End Date:", end_date)
# # # #     # Define the aggregation pipeline
# # # #     pipeline = [
# # # #         {
# # # #             '$match': {
# # # #                 'userId': ObjectId(userId),  # Match documents by userId
# # # #                 'date_time': {
# # # #                     '$gte': start_date,  # Filter documents with date_time >= startDate
# # # #                     '$lte': end_date,    # Filter documents with date_time <= endDate
# # # #                 },
# # # #             },
# # # #         },
# # # #         {
# # # #             '$project': {  # Project desired fields
# # # #                 '_id': 0,
# # # #                 'createdAt': 0,
# # # #                 'updatedAt': 0,
# # # #                 'userId': 0,
# # # #                 'mac_address_framed': 0,
# # # #             },
# # # #         },
# # # #     ]
# # # #     print("fetchinggg data")
# # # #     # Run the aggregation pipeline
# # # #     result = list(db.ecgdatas.aggregate(pipeline))  # Replace 'your_collection_name' with your actual collection name
# # # #     print("data")
# # # #     return jsonify(result)
# # import time
# # import pymongo
# # import concurrent.futures
# # from datetime import datetime
# # from bson import ObjectId
# # from pymongo import MongoClient

# # # MongoDB connection setup
# # connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
# # client = MongoClient(connection_string)
# # db = client['medicalacculive']
# # collection = db['ecgdatas']

# # # Ensure indexes on userId and date_time for faster query performance
# # collection.create_index([('userId', pymongo.ASCENDING), ('date_time', pymongo.ASCENDING)])

# # # Function to fetch data for a given time range and user ID
# # def fetch_data_chunk(userId, startTime, endTime):
# #     start_date = datetime.fromtimestamp(startTime / 1000)
# #     end_date = datetime.fromtimestamp(endTime / 1000)

# #     pipeline = [
# #         {
# #             '$match': {
# #                 'userId': ObjectId(userId),
# #                 'date_time': {
# #                     '$gte': start_date,
# #                     '$lte': end_date,
# #                 },
# #             },
# #         },
# #         {
# #             '$project': {
# #                 '_id': 0,
# #                 'createdAt': 0,
# #                 'updatedAt': 0,
# #                 'userId': 0,
# #                 'mac_address_framed': 0,
# #             },
# #         },
# #     ]

# #     try:
# #         result = list(collection.aggregate(pipeline))
# #         return result
# #     except pymongo.errors.OperationFailure as e:
# #         print(f"Error fetching data: {e}")
# #         return []  # Return empty list on error

# # # Function to fetch data in chunks using ThreadPoolExecutor
# # def fetch_data_threadpool(userId, startTime, endTime):
# #     chunk_size = 3600 * 1000  # 1 hour in milliseconds
# #     chunks = []
# #     current_time = startTime

# #     while current_time < endTime:
# #         next_time = min(current_time + chunk_size, endTime)
# #         chunks.append((userId, current_time, next_time))
# #         current_time = next_time

# #     with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
# #         results = [executor.submit(fetch_data_chunk, *chunk) for chunk in chunks]

# #         fetched_data = []
# #         for future in concurrent.futures.as_completed(results):
# #             try:
# #                 data = future.result()
# #                 if data:
# #                     fetched_data.extend(data)
# #             except Exception as e:
# #                 print(f"Error fetching data: {e}")

# #     return fetched_data

# # # Example usage
# # if __name__ == "__main__":
# #     user_id = "66277eaf7dcc5669805e807e"  # Replace with actual user ID
# #     start_time = 1713830400000  # Replace with actual start time in milliseconds
# #     end_time = 1713916800000  # Replace with actual end time in milliseconds
# #     start = time.time()
# #     data = fetch_data_threadpool(user_id, start_time, end_time)
# #     end = time.time()

# #     if data:
# #         print(f"Fetched {len(data)} records in {end - start} seconds")
# #     else:
# #         print("No records fetched")
# import time
# import pymongo
# import multiprocessing
# from datetime import datetime
# from bson import ObjectId
# from pymongo import MongoClient
# from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

# #MongoDB connection setup
# connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
# client = MongoClient(connection_string)
# db = client['medicalacculive']
# collection = db['ecgdatas']

# #Ensure indexes on userId and date_time for faster query performance
# collection.create_index([('userId', pymongo.ASCENDING), ('date_time', pymongo.ASCENDING)])

# #Function to fetch data for a given time range and user ID
# def fetch_data_chunk(args):
#     userId, startTime, endTime = args
#     start_date = datetime.fromtimestamp(startTime / 1000)
#     end_date = datetime.fromtimestamp(endTime / 1000)

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
#                 'createdAt': 0,
#                 'updatedAt': 0,
#                 'userId': 0,
#                 'mac_address_framed': 0,
#             },
#         },
#     ]

#     try:
#         result = list(collection.aggregate(pipeline))
#         return result
#     except pymongo.errors.OperationFailure as e:
#         print(f"Error fetching data: {e}")
#         return []  # Return empty list on error
# def process_ecg(ecg_signal, sample_rate):
#     cutoff_freq = 0.5
#     b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
#     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)

#     low_cutoff = 0.5
#     high_cutoff = 50
#     b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
#     ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)

#     ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)
#     out = None
#     try:
#         out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
#         r_peaks = out['rpeaks']
#         if len(r_peaks) < 2:
#             raise ValueError("Not enough beats to compute heart rate.")
#     except Exception as e:
#         print("Error:", e)
#         r_peaks = []
#     r_peaks = out['rpeaks']
#     _ ,waves_peak = nk.ecg_delineate(ecg_signal_normalized, r_peaks, sampling_rate=sample_rate, method="peak")
#     p_peaks = waves_peak['ECG_P_Peaks']
#     q_peaks = waves_peak['ECG_Q_Peaks']
#     s_peaks = waves_peak['ECG_S_Peaks']
#     t_peaks = waves_peak['ECG_T_Peaks']

#     #filtering peaks
#     p_peaks.pop()
#     q_peaks.pop()
#     s_peaks.pop()
#     t_peaks.pop()
#     p_peaks = [0 if np.isnan(x) else x for x in p_peaks]
#     q_peaks = [0 if np.isnan(x) else x for x in q_peaks]
#     s_peaks = [0 if np.isnan(x) else x for x in s_peaks]
#     t_peaks = [0 if np.isnan(x) else x for x in t_peaks]

#     ecg_signal_processed_list = ecg_signal_normalized.tolist()
#     processed_data = {
#         "ecg_vals": ecg_signal_processed_list,
#         "out": r_peaks.tolist(),
#         "r_peaks": r_peaks,
#         "p_peaks" : p_peaks,
#         "q_peaks" : q_peaks,
#         "s_peaks" : s_peaks,
#         "t_peaks" : t_peaks
#     }
#     return processed_data
# def fetch_and_process_data(chunk):
#     userId, startTime, endTime = chunk
#     result = fetch_data_chunk(chunk)
#     if result:
#         sample_rate = 200  # Example: replace with actual sampling rate
#         processed_chunks = []
#         for data_chunk in result:
#             ecg_signal = data_chunk.get('ecg_vals', [])
#             processed_data = process_ecg(ecg_signal, sample_rate)
#             processed_data["date_time"] = data_chunk['date_time']
#             processed_chunks.append(processed_data)
#         return processed_chunks
#     return None

# def fetch_data_multiprocessing(userId, startTime, endTime):
#     chunk_size = 3600 * 1000  # 1 hour in milliseconds
#     chunks = []
#     current_time = startTime

#     while current_time < endTime:
#         next_time = min(current_time + chunk_size, endTime)
#         chunks.append((userId, current_time, next_time))
#         current_time = next_time

#     with ProcessPoolExecutor(max_workers=12) as pool:
#         results = list(pool.map(fetch_and_process_data, chunks))

#     fetched_data = [item for sublist in results if sublist for item in sublist]
#     return fetched_data
# #Example usage
# if __name__ == "__main__":
#     user_id = "66277eaf7dcc5669805e807e"  # Replace with actual user ID
#     start_time = 1713830400000  # Replace with actual start time in milliseconds
#     end_time = 1713868200000  # Replace with actual end time in milliseconds
#     start = time.time()
#     data = fetch_data_multiprocessing(user_id, start_time, end_time)
#     end = time.time()
#     if data:
#         print(f"Fetched {len(data)} records in {end - start} seconds")
#     else:
#         print("No records fetched")

# # from pymongo import MongoClient
# # from bson.objectid import ObjectId

# # # MongoDB connection setup
# # connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
# # client = MongoClient(connection_string)
# # db = client['medicalacculive']
# # collection = db['users']

# # # Replace 'userId' with the actual user ID you want to query
# # userId = '66277eaf7dcc5669805e807e'

# # # Aggregation pipeline
# # pipeline = [
# #     {
# #         '$match': {
# #             '_id': ObjectId(userId),
# #         },
# #     },
# #     {
# #         '$lookup': {
# #             'from': 'doctercustomers',
# #             'localField': '_id',
# #             'foreignField': 'userId',
# #             'as': 'doctor',
# #         },
# #     },
# #     {
# #         '$unwind': {
# #             'path': '$doctor',
# #             'preserveNullAndEmptyArrays': True,
# #         },
# #     },
# #     {
# #         '$lookup': {
# #             'from': 'users',
# #             'localField': 'doctor.docterId',
# #             'foreignField': '_id',
# #             'as': 'doctorDetail',
# #         },
# #     },
# #     {
# #         '$unwind': {
# #             'path': '$doctorDetail',
# #             'preserveNullAndEmptyArrays': True,
# #         },
# #     },
# #     {
# #         '$project': {
# #             'first_Name': 1,
# #             'last_Name': 1,
# #             'DOB': 1,
# #             'gender': 1,
# #             'countryCode': 1,
# #             'mobile_no': 1,
# #             'imageUrl': 1,
# #             'email': 1,
# #             'nationality': 1,
# #             'address': '$Address',
# #             'height': 1,
# #             'weight': 1,
# #             'medical_history': 1,
# #             'emergencyContacts': 1,
# #             'doctorDetail': {
# #                 'first_Name': '$doctorDetail.first_Name',
# #                 'last_Name': '$doctorDetail.last_Name',
# #                 'DOB': '$doctorDetail.DOB',
# #                 'email': '$doctorDetail.email',
# #                 'gender': '$doctorDetail.gender',
# #                 'countryCode': '$doctorDetail.countryCode',
# #                 'mobile_no': '$doctorDetail.mobile_no',
# #                 'address': '$doctorDetail.Address',
# #                 'imageUrl': '$doctorDetail.imageUrl',
# #                 'licenseDetail': '$doctorDetail.licenseDetail',
# #             },
# #         },
# #     },
# # ]

# # # Execute the aggregation pipeline
# # result = list(collection.aggregate(pipeline))

# # # Print the result
# # for doc in result:
# #     print(doc)
