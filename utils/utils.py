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
from functools import wraps
import asyncio
import aiohttp
from aiohttp import ClientSession
from flask import Flask, make_response, request, jsonify
import time
import pymongo
import multiprocessing
from datetime import datetime, timezone
from bson import ObjectId
from pymongo import MongoClient
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from scipy import signal
import neurokit2 as nk
import pymongo
from bson import ObjectId
from datetime import datetime, timezone
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import motor.motor_asyncio
# import nest_asyncio
# connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
# client = MongoClient(connection_string)
# db = client['medicalacculive']
# collection = db['ecgdatas']
connection_string = "mongodb://root:Dfsdag7dgrwe3whfd@194.233.69.96:27017/medicalacculive?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false"
client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
db = client['medicalacculive']
collection = db['ecgdatas']
  
collection2 = db['users']
#collection.create_index([('userId', pymongo.ASCENDING), ('date_time', pymongo.ASCENDING)])

# nest_asyncio.apply() 
def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter(), time.process_time()
        result = func(*args, **kwargs)
        t2 = time.perf_counter(), time.process_time()
        
        real_time = t2[0] - t1[0]
        cpu_time = t2[1] - t1[1]
        
        # Open the CSV file and write the data
        with open('function_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty to write the header
            if file.tell() == 0:
                writer.writerow(['Function Name', 'Real Time', 'CPU Time'])
            writer.writerow([func.__name__, f"{real_time:.2f}", f"{cpu_time:.2f}"])
        
        return result
    return wrapper
# def timer_decorator(func):
#     def wrapper(*args, **kwargs):
#         t1 = time.perf_counter(), time.process_time()
#         result = func(*args, **kwargs)
#         t2 = time.perf_counter(), time.process_time()
#         print(f"{func.__name__}()")
#         print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
#         print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
#         print()
#         return result
#     return wrapper
# def timer_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         if execution_time > 0.0:
#             print(f"Execution time of {func.__name__}: {execution_time} seconds")
#         return result
#     return wrapper

@timer_decorator
def detect_trigeminy(rr_intervals):
    # Check for trigeminy pattern based on RR intervals
    temp_count = 0
    length = len(rr_intervals)
    for i in range(1, length - 1, 3):
       #if 0.75 * rr_intervals[i-1] >  rr_intervals[i] and 0.60 * rr_intervals[i+1] >  rr_intervals[i]:  #change threshold val accordingly
        if rr_intervals[i] < 0.75 * rr_intervals[i - 1] and rr_intervals[i] < 0.60 * rr_intervals[i + 1]:
            temp_count += 1
            if temp_count > 1:
                return True
    return False

@timer_decorator    
def detect_bigeminy(rr_intervals):  
    # Check for bigeminy pattern based on RR intervals
    # if len(rr_intervals) % 2 == 0 and all(rr_intervals[i] < 0.75 * np.mean(rr_intervals) for i in range(1, len(rr_intervals), 2)):   # changed it from 0.75 to 0.65
    #     return True
    # return False
    length = len(rr_intervals)
    if length % 2 != 0:
        return False
    mean_rr = np.mean(rr_intervals)
    threshold = 0.65 * mean_rr
    for i in range(1, length, 2):
        if rr_intervals[i] >= threshold:
            return False
    
    return True

@timer_decorator
def detect_atrial_flutter(pPinterval):
    temp_count = 0
    mean_pP = np.mean(pPinterval)
    for p_p_interval in pPinterval:
        # Calculate the absolute deviation from the mean and check if it exceeds 15%
        if abs(p_p_interval - mean_pP) / mean_pP > 0.15:
            temp_count += 1
            if temp_count > 2:
                return True
    return False
@timer_decorator
def detect_atrioventricular_block(pr_intervals):    # Second degree and Third degree is left
    temp_count = 0
    for pr_interval in pr_intervals:
        if pr_interval > 0.2:
            temp_count +=1
            if temp_count > 2:
                return True
    return False

@timer_decorator
def calculate_intervals(r_peaks,p_peaks,q_peaks,s_peaks,t_peaks):
    # Calculates all the intervals
    rRinterval=[]
    rRsquare=[]
    qSinterval=[]
    rTinterval=[]
    pQinterval=[]
    rSinterval=[]
    sTinterval=[]
    qRinterval=[]
    qTinterval=[]
    pRinterval=[]
    QRSinterval=[]
    pPinterval=[]

    for i in range(1,len(r_peaks)):
        rRinterval.append((r_peaks[i]-r_peaks[i-1])/200)

    for i in range(1,len(p_peaks)):
        pPinterval.append((p_peaks[i]-p_peaks[i-1])/200)

    for i in range(1,len(rRinterval)):
        rRsquare.append(np.square((rRinterval[i])-(rRinterval[i-1])))

    for i in range(min(len(s_peaks),len(q_peaks))):
        if (s_peaks[i]!=0 and q_peaks[i]!=0):
            qSinterval.append((s_peaks[i]-q_peaks[i])/200)

    for i in range(min(len(t_peaks),len(r_peaks))):
        if (t_peaks[i]!=0 and r_peaks[i]!=0):
            rTinterval.append((t_peaks[i]-r_peaks[i])/200)

    for i in range(min(len(p_peaks),len(q_peaks))):
        if (p_peaks[i]!=0 and q_peaks[i]!=0):
            pQinterval.append((q_peaks[i]-p_peaks[i])/200)

    for i in range(min(len(s_peaks),len(r_peaks))):
        if (s_peaks[i]!=0 and r_peaks[i]!=0):
            rSinterval.append((s_peaks[i]-r_peaks[i])/200)

    for i in range(min(len(s_peaks),len(t_peaks))):
        if (s_peaks[i]!=0 and t_peaks[i]!=0):
            sTinterval.append((t_peaks[i]-s_peaks[i])/200)

    for i in range(min(len(r_peaks),len(q_peaks))):
        if (r_peaks[i]!=0 and q_peaks[i]!=0):
            qRinterval.append((r_peaks[i]-q_peaks[i])/200)

    for i in range(min(len(q_peaks),len(t_peaks))):
        if (t_peaks[i]!=0 and q_peaks[i]!=0):
            qTinterval.append((t_peaks[i]-q_peaks[i])/200)

    for i in range(min(len(p_peaks),len(r_peaks))):
        if (p_peaks[i]!=0 and r_peaks[i]!=0):
            pRinterval.append((r_peaks[i]-p_peaks[i])/200)

    for i in range(min(len(qRinterval),len(rSinterval))):
        QRSinterval.append(qRinterval[i]+rSinterval[i])

    return rRinterval,pPinterval,rRsquare,qSinterval,rTinterval,pQinterval,rSinterval,sTinterval,qRinterval,qTinterval,pRinterval,QRSinterval

@timer_decorator
def detect_afib(rr_intervals, p_peaks, p_peak_threshold=6):
    mean_rr_interval = np.mean(rr_intervals)
    rr_regular = all(np.abs(rr_interval - mean_rr_interval) < 0.1 * mean_rr_interval for rr_interval in rr_intervals)
    if p_peaks.count(0) >= p_peak_threshold and rr_regular == False:
        return True
    else : return False
@timer_decorator
def detect_vt(qrs_mean, st_mean, heart_rate, p_peaks):
    qrs_mean_threshold = 0.12
    st_segment_threshold = 0.12
    rapid_heart_rate_threshold = 100
    if qrs_mean_threshold > 0.12:
        if st_mean > st_segment_threshold:
            if p_peaks.count(0) > 2:
                if heart_rate > rapid_heart_rate_threshold:
                    return True
    return False
@timer_decorator
def detect_svt(qrs_mean, heart_rate, p_peaks):
    qrs_mean_threshold = 0.12
    rapid_heart_rate_threshold = 100
    if qrs_mean < 0.12:
        if p_peaks.count(0) > 2:
            if heart_rate > rapid_heart_rate_threshold:
                return True
    return False
@timer_decorator
def filter_ecg_data_around_timestamp(timestamp, json_data, window_size=1):
    ecg_data_filtered = []
    timestamps = [data_point["date_time"] for data_point in json_data]
    index = timestamps.index(timestamp)
    start_index = max(0, index - window_size)
    end_index = min(len(json_data), index + window_size + 1)

    # If there is no data before the timestamp, include the first packet along with two subsequent packets
    if index < window_size:
        start_index = 0
        end_index = min(2 * window_size + 1, len(json_data))
    # If there is no data after the timestamp, include the two packets before it
    elif index > len(json_data) - window_size - 1:
        start_index = max(0, len(json_data) - 2 * window_size - 1)
        end_index = len(json_data)

    for data_point in json_data[start_index:end_index]:
        ecg_data_filtered.append({"timestamp": data_point["date_time"],
                                  "ecg_vals": data_point["ecg_vals"]})
    return ecg_data_filtered
    
@timer_decorator
async def fetch_user_data(user_id):
    # Define the API endpoint
    # Define the API endpoint
    pipeline = [
    {
        '$match': {
            '_id': ObjectId(user_id),
        },
    },
    {
        '$lookup': {
            'from': 'doctercustomers',
            'localField': '_id',
            'foreignField': 'userId',
            'as': 'doctor',
        },
    },
    {
        '$unwind': {
            'path': '$doctor',
            'preserveNullAndEmptyArrays': True,
        },
    },
    {
        '$lookup': {
            'from': 'users',
            'localField': 'doctor.docterId',
            'foreignField': '_id',
            'as': 'doctorDetail',
        },
    },
    {
        '$unwind': {
            'path': '$doctorDetail',
            'preserveNullAndEmptyArrays': True,
        },
    },
    {
        '$project': {
            'first_Name': 1,
            'last_Name': 1,
            'DOB': 1,
            'gender': 1,
            'countryCode': 1,
            'mobile_no': 1,
            'imageUrl': 1,
            'email': 1,
            'nationality': 1,
            'address': '$Address',
            'height': 1,
            'weight': 1,
            'medical_history': 1,
            'emergencyContacts': 1,
            'doctorDetail': {
                'first_Name': '$doctorDetail.first_Name',
                'last_Name': '$doctorDetail.last_Name',
                'DOB': '$doctorDetail.DOB',
                'email': '$doctorDetail.email',
                'gender': '$doctorDetail.gender',
                'countryCode': '$doctorDetail.countryCode',
                'mobile_no': '$doctorDetail.mobile_no',
                'address': '$doctorDetail.Address',
                'imageUrl': '$doctorDetail.imageUrl',
                'licenseDetail': '$doctorDetail.licenseDetail',
            },
        },
    },
]
# Execute the aggregation pipeline
    cursor = collection2.aggregate(pipeline)
    result = await cursor.to_list(length=None) 
    user_data = result[0] # Extract the 'data' part of the response
    referring_doctor_first_name = user_data["doctorDetail"].get("first_Name", "").capitalize()
    referring_doctor_last_name = user_data["doctorDetail"].get("last_Name", "").capitalize()
    referring_doctor = referring_doctor_first_name + " " + referring_doctor_last_name
    dob = user_data["DOB"]
    current_date = datetime.now()
    age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))

    if not referring_doctor.strip():  # If referring doctor data is empty
                referring_doctor = "Self"
    return {
                "patient_name": user_data["first_Name"].capitalize() + " " + user_data["last_Name"].capitalize(),
                "referring_doctor": referring_doctor,
                "gender": user_data["gender"].capitalize(),
                "_id": user_data["medical_history"]["_id"],
                "mobile_no": user_data["mobile_no"],
                "age" : age
            }
    

# @timer_decorator
# def fetch_ecg_data(userId, startDate, endDate):
#     # Define the API endpoint
#     api_endpoint = f"https://api.accu.live/api/v1/devices/getecgdata?userId={userId}&startDate={startDate}&endDate={endDate}"
    
#     try:
#         # Make a GET request to the API
#         response = requests.get(api_endpoint)
#         # Check if the request was successful (status code 200)
#         if response.status_code == 200:
#             # Return the JSON response
#             return response.json()
#         else:
#             # If the request was not successful, raise an exception
#             response.raise_for_status()
#     except requests.RequestException as e:
#         # Handle any request exceptions (e.g., connection errors, timeout)
#         print("Error fetching data from the API:", e)
#         return None
@timer_decorator
def generate_image_around(ecg_data_hr,hr_datetime):
    plt.figure(figsize=(10, 1))
    # Concatenate all ecg_vals packets into a single list
    concatenated_ecg_values = [value for data_point in ecg_data_hr for value in data_point["ecg_vals"]]
    # Plot all ecg_vals as a continuous line
    plt.plot(concatenated_ecg_values, color='black', linewidth=1)
    # Highlight the portion corresponding to the maximum heart rate packet
    highlight_start = 0
    for data_point in ecg_data_hr:
        timestamp = data_point["timestamp"]
        ecg_values = data_point["ecg_vals"]
        length = len(ecg_values)
        if timestamp == hr_datetime:
            plt.axvspan(highlight_start, highlight_start + length, color='lightblue', alpha=0.5)
            break  # Highlight only the first occurrence
        highlight_start += length
    # Save the plot as an image
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.legend().remove()
    plt.margins(x=0)
    return plt
@timer_decorator
def generate_ecg_image(data):
    # Extract data from data dictionary
    bg = os.path.join('static/images', 'bg_plot.png')
    r_peaks = data["r_peaks"]
    rRinterval = data["rRinterval"]
    ecg_signal_normalized = data["ecg_signal_normalized"]
    plt.figure(figsize=(20, 2))
    image = plt.imread(bg)
    plt.imshow(image, aspect='auto', extent=[0, len(ecg_signal_normalized) + 1, np.min(ecg_signal_normalized) - 1, np.max(ecg_signal_normalized) + 1])
    plt.plot(ecg_signal_normalized, color='black', label='ECG Signal',linewidth=1)
    length = len(r_peaks)
    # Add "N" labels on top of each peak with rR interval values
    for i, peak_index in enumerate(r_peaks):
        rR_value = rRinterval[i % len(rRinterval)]  # Get corresponding rR interval value
        
        # Calculate the x-coordinate for positioning the rR interval value between two consecutive peaks
        if i < length - 1:
            next_peak_index = r_peaks[i + 1]
            midpoint = (peak_index + next_peak_index) / 2
        else:
            midpoint = peak_index
        
        # Add "N" label above the peak
        plt.text(peak_index, np.max(ecg_signal_normalized) + 0.8, 'N', fontsize=12, ha='center', va='top', color='black')
        
        # Add rR interval value between two consecutive "N" labels and center it
        if i < length - 1:
            plt.text(midpoint, np.max(ecg_signal_normalized) + 0.8, f'{int(rR_value * 1000)}', fontsize=12, ha='center', va='top', color='black')
    
    # Configure plot settings
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.legend().remove()
    plt.margins(x=0)
    
    # Return the plot object
    return plt
@timer_decorator
def get_max_interval(pauses_data):
    max_rr_data = None
    max_rr = -float('inf')
    for data in pauses_data:
      temp = max(data["rRinterval"])
      if temp > max_rr :
        max_rr = temp
        max_rr_data = {
            "timestamp":data["timestamp"],
            "heartbeats": data["heartbeats"],
            "r_peaks": data["r_peaks"],
            "rRinterval": data["rRinterval"],
            "ecg_signal_normalized": data["ecg_signal_normalized"],
            "duration" : data["duration"]
        }
    return max_rr_data, max_rr
@timer_decorator
def convert_to_local(timestamp):
    timestamp_datetime = timestamp
    #timestamp_datetime = datetime.fromisoformat(timestamp[:-1])
    #local_tz = datetime.now(pytz.timezone('UTC')).astimezone().tzinfo
    local_offset = timedelta(hours=5, minutes=30)
    timestamp_local = timestamp_datetime + local_offset
    timestamp_local_str = timestamp_local.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp_local_str
@timer_decorator
def calculate_start(pause_length,afib_length,ischemia_length,pvc_length,vt_length,af_length,atv_length,svt_length,bigeminy_length):
    afib_start = pause_length // 4 + (4 if pause_length % 4 == 3 else 3)
    pvc_start = afib_length // 4 + (afib_start + 1  if afib_length % 4 == 3 else afib_start)
    ischemia_start = pvc_length // 4 + (pvc_start + 1  if pvc_length % 4 == 3 else pvc_start)
    vt_start = ischemia_length // 4 + (ischemia_start + 1  if ischemia_length % 4 == 3 else ischemia_start)
    af_start = vt_length // 4 + (vt_start + 1  if vt_length % 4 == 3 else vt_start)
    atv_start = af_length // 4 + (af_start + 1  if af_length % 4 == 3 else af_start)
    svt_start = atv_length // 4 + (atv_start + 1  if atv_length % 4 == 3 else atv_start)
    bigeminy_start = svt_length // 4 + (svt_start + 1  if svt_length % 4 == 3 else svt_start)
    trigeminy_start = bigeminy_length // 4 + (bigeminy_start + 1  if bigeminy_length % 4 == 3 else bigeminy_start)
    return afib_start,pvc_start,ischemia_start,vt_start,af_start,atv_start,svt_start,bigeminy_start,trigeminy_start
@timer_decorator
def add_footer(canvas, page_num):
    canvas.setFont("Helvetica", 9)
    header_height = 6
    canvas.setFillColor(colors.HexColor('#545353'))
    canvas.drawString(20,letter[1] - header_height*120-40,"Contact number - 079 4003 7674  https://www.bacancytechnology.com/ Copyright ©2024 BACANCY. All Rights Reserved")
    canvas.drawString(550,letter[1] - header_height*120-40,"Page "+str(page_num))
@timer_decorator
def add_header(canvas, patient_name, gender, age, pid):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor('#545353'))
    header_height = 6
    canvas.drawString(20,letter[1] - header_height*5,f"{patient_name}     {gender[:1].capitalize()}   {age} Yr ")
    canvas.drawString(460,letter[1] - header_height*5,f"ID {pid}")
    canvas.setLineWidth(2)
    canvas.line(20, letter[1] - header_height*15, letter[0] - 20, letter[1] - header_height*15)
    canvas.setLineWidth(1)
@timer_decorator
def add_heading(canvas, x, y, width, height, heading_text, padding = 10):
    canvas.setFillColor(colors.lightgrey)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.rect(x, y, width, height, stroke=0, fill=1)   # Draw the border around the box
      # Draw upper border only
    canvas.line(x, y + height, x + width, y + height)
      # Draw lower border only
    canvas.line(x, y, x + width, y)
      # Calculate the coordinates for centering the text
    text_x = x + padding
    text_y = y - 6 + (height) / 2
    canvas.setFillColorRGB(0, 0, 0)
    # Draw the heading "Pause" centered inside the box
    canvas.drawString(text_x, text_y, heading_text)
@timer_decorator
def add_images(canvas, data, start_y, disease, i, patient_name, gender, age, pid, json_data):
    img_height = 70
    img_around_height = 40
    blank_box_width = (letter[0] - 40)
    header_height = 6
    start_y = start_y
    for entry in data:
        required_height = img_height + img_around_height + 45  # Adjust as needed
        if start_y - required_height < 8:
            page_num = canvas.getPageNumber()
            add_footer(canvas,page_num)
            canvas.showPage()
            add_header(canvas, patient_name, gender, age, pid)
            start_y = letter[1] - header_height * 10 - 75

        imgdata = BytesIO()
        plt = generate_ecg_image(entry)  # Adjust as needed
        plt.savefig(imgdata, format='png', bbox_inches='tight', pad_inches=0)
        imgdata.seek(0)
        img = ImageReader(imgdata)
        plt.close()
        #timestamp = datetime.fromisoformat(str(entry["timestamp"][:-1]))  # Adjust as needed
        timestamp = entry["timestamp"]  # Adjust as needed
        
        ecg_data = filter_ecg_data_around_timestamp(entry["timestamp"], json_data)  # Adjust as needed
        timestamp_local = convert_to_local(entry["timestamp"])  # Adjust as needed
        imgdata = BytesIO()
        plt = generate_image_around(ecg_data, timestamp)
        plt.savefig(imgdata, format='png',bbox_inches='tight', pad_inches=0)
        imgdata.seek(0)  # Reset the pointer to the beginning of the BytesIO object
        img_around = ImageReader(imgdata)
        plt.close()

        ecg_rate_data = [
                [f"{disease}","Duration","Heart Rate","Activities","PTE"],
                [f"{timestamp_local}",f"{entry['duration']} sec",f"{entry['heartbeats']} bpm","No", "No"],
            ]
        num_columns = len(ecg_rate_data[0])
        column_widths = [(blank_box_width / num_columns)-8] * num_columns
        heart_rate_table = Table(ecg_rate_data, colWidths=column_widths, rowHeights=20)
        heart_rate_table.setStyle(TableStyle([
                ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header row text color
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),       # Center alignment for all cells
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),  # Background color for all cells
                ('PADDING', (0, 0), (-1, -1), (5, 10)),       # Padding for all cells
            ]))
        heart_rate_table.wrapOn(canvas, 0, 0)
        heart_rate_table.drawOn(canvas, 60, start_y)
        # Draw img and img_around
        canvas.drawImage(img, 25, start_y - 72, width=567, height=img_height)
        canvas.drawImage(img_around, 25, start_y - img_height - 42, width=567, height=img_around_height)
        canvas.setFillColorRGB(0, 0, 0)  # Black color
        canvas.circle(40, start_y + 18, 13, stroke=1, fill=1)  # Draw black circle
        canvas.setFillColorRGB(1, 1, 1)  # White color
        canvas.setFont("Helvetica", 12)
        if i > 9 and i < 100:
          canvas.drawString(33, start_y + 13, f"{i}")
        elif i > 99:
          canvas.drawString(30, start_y + 13, f"{i}")
        else:   
          canvas.drawString(37, start_y + 13, f"{i}")
        i += 1
        # Update start_y for the next iteration
        start_y -= (img_height + img_around_height + 45)
    return i, start_y
@timer_decorator
def get_first_image(canvas, svt_data_start, page_start, y):
    x = 250
    box_width = 340
    box_height = 50
    canvas.drawString(x + 300, y - 15, f"Page {page_start}" )
    canvas.linkURL(f"#page={page_start}", (x + 300, y - 15, x + 340, y + 2))
    canvas.setFillColorRGB(1, 1, 1)  # White color
    canvas.rect(x, y-20, box_width, -box_height, stroke=1, fill=1)
    timestamp_local = convert_to_local(svt_data_start["timestamp"])
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(x + 75, y - 15, f'First detected at: {timestamp_local}' )
    imgdata = BytesIO()
    plt = generate_ecg_image(svt_data_start)
    plt.savefig(imgdata, format='png', bbox_inches='tight', pad_inches=0)
    imgdata.seek(0)
    img_vt = ImageReader(imgdata)
    plt.close()
    canvas.drawImage(img_vt, x, y-20, width=box_width , height=-box_height)
    y -= box_height + 1
    return y
@timer_decorator
def generate_plot(hours, heartbeats_list):
    plt.figure(figsize=(10, 2))
    plt.plot(hours, heartbeats_list, linestyle='-', color='black',linewidth=0.5)
    plt.title('Heartbeats Over Time (0-23 hours)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Heartbeats (BPM)')
    plt.xticks(range(25))
    plt.tight_layout()
    return plt

@timer_decorator
def create_indexes():
    collection.create_index([('userId', 1), ('date_time', 1)])

# # Function to fetch data for a given time range and user ID
# @timer_decorator
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
#                 'createdAt': 0,
#                 'updatedAt': 0,
#                 'userId': 0,
#                 'mac_address_framed': 0,
#             },
#         },
#     ]

#     try:
#         cursor = collection.aggregate(pipeline)
#         result = list(cursor)
#         return result
#     except Exception as e:
#         print(f"Error fetching data: {e}")
#         return []  # Return empty list on error

# # Process ECG data chunk
# @timer_decorator
# def process_ecg_data_chunk(data_chunk, sample_rate):
#     ecg_signal = data_chunk.get('ecg_vals', [])
    
#     # Example processing steps (filters and peaks extraction)
#     processed_data = process_ecg(ecg_signal, sample_rate)
#     processed_data["date_time"] = data_chunk['date_time']
#     return processed_data

# # Example ECG processing (replace with actual processing logic)
# @timer_decorator
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
#     out = None
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
#         "out": r_peaks.tolist(),
#         "r_peaks": r_peaks,
#         "p_peaks": p_peaks,
#         "q_peaks": q_peaks,
#         "s_peaks": s_peaks,
#         "t_peaks": t_peaks
#     }
#     print("k")
#     return processed_data
# @timer_decorator
# async def fetch_and_process_data(chunk):
#     userId, startTime, endTime = chunk
#     result = await fetch_data_chunk(userId, startTime, endTime)
    
#     if result:
#         sample_rate = 200  # Replace with actual sampling rate
#         processed_chunks = [process_ecg_data_chunk(data_chunk, sample_rate) for data_chunk in result]
#         return processed_chunks
#     return []
# @timer_decorator
# async def fetch_data_async(userId, startTime, endTime):
#     chunk_size = 1800 * 1000  # Adjust chunk size as needed
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
        result = await cursor.to_list(length=None)  # Fetch all results at once
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def process_ecg_data_chunk(data_chunk, sample_rate):
    ecg_signal = data_chunk.get('ecg_vals', [])

    # High-pass filter
    cutoff_freq = 0.5
    b, a = signal.butter(2, cutoff_freq / (0.5 * sample_rate), 'high')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal)

    # Band-pass filter
    low_cutoff = 0.5
    high_cutoff = 50
    b, a = signal.butter(2, [low_cutoff / (0.5 * sample_rate), high_cutoff / (0.5 * sample_rate)], 'band')
    ecg_signal_filt = signal.filtfilt(b, a, ecg_signal_filt)

    # Normalize the signal
    ecg_signal_normalized = (ecg_signal_filt - np.mean(ecg_signal_filt)) / np.std(ecg_signal_filt)

    try:
        out = ecg.ecg(signal=ecg_signal_normalized, sampling_rate=sample_rate, show=False)
        r_peaks = out['rpeaks']
    except Exception as e:
        print("Error in ECG processing:", e)
        r_peaks = []

    _, waves_peak = nk.ecg_delineate(ecg_signal_normalized, r_peaks, sampling_rate=sample_rate, method="peak")

    def handle_nan(peaks):
        return [0 if np.isnan(x) else x for x in peaks[:-1]]

    p_peaks = handle_nan(waves_peak['ECG_P_Peaks'])
    q_peaks = handle_nan(waves_peak['ECG_Q_Peaks'])
    s_peaks = handle_nan(waves_peak['ECG_S_Peaks'])
    t_peaks = handle_nan(waves_peak['ECG_T_Peaks'])

    ecg_signal_processed_list = ecg_signal_normalized.tolist()
    processed_data = {
        "ecg_vals": ecg_signal_processed_list,
        "out": r_peaks.tolist(),
        "r_peaks": r_peaks,
        "p_peaks": p_peaks,
        "q_peaks": q_peaks,
        "s_peaks": s_peaks,
        "t_peaks": t_peaks,
        "date_time": data_chunk['date_time']
    }
    print("k")
    return processed_data

async def fetch_and_process_data(userId, startTime, endTime, sample_rate):
    chunk_size = 1800 * 1000  # 1800 seconds in milliseconds (30 minutes)
    current_time = int(startTime)
    tasks = []
    endTime = int(endTime)

    while current_time < endTime:
        chunk_end_time = min(current_time + chunk_size, endTime)
        tasks.append(fetch_data_chunk(userId, current_time, chunk_end_time))
        current_time += chunk_size

    # Execute tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Flatten the list of results
    data_chunks = [item for sublist in results for item in sublist]

    # Process data chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(executor, process_ecg_data_chunk, chunk, sample_rate) for chunk in data_chunks]
        processed_chunks = await asyncio.gather(*futures)

    return processed_chunks

async def fetch_data_async(userId, startTime, endTime):
    sample_rate = 200  # Replace with actual sampling rate
    mid_point = (int(endTime) + int(startTime)) // 2
    
    # Divide into two parts
    tasks = [
        fetch_and_process_data(userId, startTime, mid_point, sample_rate),
        fetch_and_process_data(userId, mid_point, endTime, sample_rate)
    ]
    
    # Execute both parts concurrently
    results = await asyncio.gather(*tasks)
    
    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]
    
    return flattened_results
@timer_decorator
async def generate_pdf(userId, startDate, endDate):
  i=0
  logo = os.path.join('static/images', 'Full Logo - On Light.png')
  pdf_buffer = BytesIO()
  sample_rate = 200
  min_data, max_data  = [],[]
  heart_rate_data = {}
  num_data_points = 0
  total_mean_rR = 0
  total_mean_pP = 0
  total_mean_qS = 0
  total_mean_rT = 0
  total_mean_pQ = 0
  total_mean_rS = 0
  total_mean_sT = 0
  total_mean_qR = 0
  total_mean_qT = 0
  total_mean_pR = 0
  total_mean_qrs = 0
  total_mean_qtc = 0
  sinus_bradycardia_count = 0
  max_hr=0
  min_hr=0
  pause = False
  afib = False
  pvc = False
  svt = False
  heartbeats = 0
  all_ecg_signals_np = np.array([])
  pauses_data, pvc_data, myocardial_ischemia_data, afib_data, vt_data, svt_data, bigeminy_data, atrioventricular_data, af_data, trigeminy_data, disease_list = [],[],[],[],[],[],[],[],[],[],[]
  #create_indexes()
    
  json_data = await fetch_data_async(userId, startDate, endDate)
  
 
  for data_point in json_data:
    date_time = data_point["date_time"]

    out = data_point["out"]
    if out == None:
        continue
    ecg_signal_normalized = data_point["ecg_vals"]
    
    r_peaks = data_point['r_peaks']
    p_peaks = data_point['p_peaks']
    q_peaks = data_point['q_peaks']
    s_peaks = data_point['s_peaks']
    t_peaks = data_point['t_peaks']
    rRinterval,pPinterval,rRsquare,qSinterval,rTinterval,pQinterval,rSinterval,sTinterval,qRinterval,qTinterval,pRinterval,QRSinterval= calculate_intervals(r_peaks,p_peaks,q_peaks,s_peaks,t_peaks)

      # Get the starting and ending indices of the appended data
    start_index = (len(all_ecg_signals_np) - len(ecg_signal_normalized)) + 1
    end_index = len(all_ecg_signals_np)
    duration = round((end_index/200) - (start_index/200), 2)


      #calculate mean of all intervals
    rms_rR=np.sqrt(np.nanmean(rRsquare) if len(rRsquare) > 0 else 0)*1000
    mean_rR = np.nanmean(rRinterval) if len(rRinterval) > 0 else 0
    mean_pP=np.nanmean(pPinterval) if len(pPinterval) > 0 else 0
    mean_qS=np.nanmean(qSinterval) if len(qSinterval) > 0 else 0
    mean_rT=np.nanmean(rTinterval) if len(rTinterval) > 0 else 0
    mean_pQ=np.nanmean(pQinterval) if len(pQinterval) > 0 else 0
    mean_rS=np.nanmean(rSinterval) if len(rSinterval) > 0 else 0
    mean_sT=np.nanmean(sTinterval) if len(sTinterval) > 0 else 0
    mean_qR=np.nanmean(qRinterval) if len(qRinterval) > 0 else 0
    mean_qT=np.nanmean(qTinterval) if len(qTinterval) > 0 else 0
    mean_pR=np.nanmean(pRinterval) if len(pRinterval) > 0 else 0
    mean_qrs=np.nanmean(QRSinterval) if len(QRSinterval) > 0 else 0

    total_mean_rR += mean_rR
    total_mean_pP += mean_pP
    total_mean_qS += mean_qS
    total_mean_rT += mean_rT
    total_mean_pQ += mean_pQ
    total_mean_rS += mean_rS
    total_mean_sT += mean_sT
    total_mean_qR += mean_qR
    total_mean_qT += mean_qT
    total_mean_pR += mean_pR
    total_mean_qrs += mean_qrs
    total_mean_qtc += (mean_qT/np.sqrt(mean_rR))
    num_data_points += 1
      # Calculate HR
    heartbeats = round(60/mean_rR)
    event_data = {
                "timestamp": date_time,
                "heartbeats": heartbeats,
                "r_peaks": r_peaks,
                "rRinterval": rRinterval,
                "ecg_signal_normalized": ecg_signal_normalized,
                "duration" : duration
            }
    if heartbeats < 60:
        sinus_bradycardia_count +=1

    if detect_bigeminy(rRinterval):
        bigeminy_data.append(event_data)
    if detect_trigeminy(rRinterval):
        trigeminy_data.append(event_data)
    if all(0 <= val <= 0.9 for val in pPinterval) and detect_atrial_flutter(pPinterval):
        af_data.append(event_data)
    if detect_atrioventricular_block(pRinterval):
        atrioventricular_data.append(event_data)

    temp_count_ischemia = 0
    temp_count_pvc = 0
    for i in s_peaks:
        if (ecg_signal_normalized[i] <= -1.5) and (ecg_signal_normalized[i] >= -5):
          temp_count_ischemia += 1
        if (ecg_signal_normalized[i] < -5):
          temp_count_pvc +=2

    if all_ecg_signals_np.size == 0:
          all_ecg_signals_np = np.array(ecg_signal_normalized)
    else:
          all_ecg_signals_np = np.concatenate((all_ecg_signals_np, ecg_signal_normalized))

      # Store HeartRate with timestamp
    heart_rate_data[date_time] = heartbeats
    is_vt_detected = detect_vt(QRSinterval, mean_sT, heartbeats, p_peaks)
    if is_vt_detected:
        vt_data.append(event_data)
    svt = detect_svt(mean_qrs, heartbeats, p_peaks)
    if svt:
        svt_data.append(event_data)
    afib = detect_afib(rRinterval, p_peaks)
    if afib:
        afib_data.append(event_data)
    if any(rr_interval * 1000 > 2000 for rr_interval in rRinterval):
        pause = True
        pauses_data.append(event_data)
    if temp_count_ischemia > 0:
        myocardial_ischemia_data.append(event_data)
    if temp_count_pvc > 0:
        pvc = True
        pvc_data.append(event_data)
    if max_hr != 0:
        if heartbeats < min_hr:
            min_data.clear()  # Clear the list
            min_data.append(event_data)
            min_hr = heartbeats
        if heartbeats > max_hr:
            max_data.clear()  # Clear the list
            max_data.append(event_data)
            max_hr = heartbeats
    else:
        min_hr = heartbeats  # Clear the list
        min_data.append(event_data)
        max_hr = heartbeats
        max_data.append(event_data)
    
  overall_mean_rR = round((total_mean_rR / num_data_points) * 1000)
  overall_mean_pP = round((total_mean_pP / num_data_points) * 1000)
  overall_mean_pQ = round((total_mean_pQ / num_data_points) * 1000)
  overall_mean_sT = round((total_mean_sT / num_data_points) * 1000)
  overall_mean_qT = round((total_mean_qT / num_data_points) * 1000)
  overall_mean_pR = round((total_mean_pR / num_data_points) * 1000)
  overall_mean_qrs = round((total_mean_qrs / num_data_points) * 1000)
  overall_mean_qtc = round((total_mean_qtc / num_data_points) * 1000)

  timestamps = [timestamp for timestamp in heart_rate_data.keys()]
  reading_time = timestamps[-1] - timestamps[0]
  difference_in_hours = reading_time.total_seconds() / 3600
  reading_time = round(difference_in_hours, 2)
  heartbeats_list = list(heart_rate_data.values())
  overall_mean_heartbeats = round(np.mean(heartbeats_list))
  offset = timedelta(hours=5, minutes=30)
  timestamps_adjusted = [timestamp + offset for timestamp in timestamps]
  # Convert each adjusted timestamp to hours with fractions
  hours = [timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600 for timestamp in timestamps_adjusted]
  # Plot heartbeats for 24 hrs
  #return {"d":"s"}
  #-------------------- PDF -------------------#
  pdf_filename = "ecg_report.pdf"
  canvas = Canvas(pdf_buffer, pagesize=letter)

  canvas.drawImage(logo, 20, letter[1] - 60, width=150, height=40)
  header_height = 6  # Header height in cm

  # Add heading
  canvas.setFillColor(colors.black)
  canvas.setFont("Helvetica-Bold", 20)
  canvas.drawString(400, letter[1] - 55, "ECG REPORT")
  canvas.line(20, letter[1] - header_height*10-10, letter[0] - 20, letter[1] - header_height*10-10)
  patient_data = await fetch_user_data(userId)

  ref_doc= patient_data["referring_doctor"]
  gender = patient_data["gender"]
  patient_name = "Mr "+ patient_data["patient_name"] if gender == "Male" else "Ms " + patient_data["patient_name"]
  pid = patient_data["_id"]
  age = patient_data["age"]
  mobile_no = patient_data["mobile_no"]
  record_for = json_data[0]["date_time"]
  formatted_date_record_for = record_for.strftime("%y/%m/%d")


  data = [
      [f"Name: {patient_name}","","","","","",f"Referal Doctor: {ref_doc}"],
      [f"Age: {age} Yr        Gender: {gender}","","","","","",f"ID: {pid}","",f"Tel: {mobile_no}"],
      [f"Record For: {formatted_date_record_for}"],
      ["Complaints: Cardiac Arrhythmia"]
  ]
  
  # Create table and set style
  table = Table(data)
  table.setStyle(TableStyle([
      ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header row text color
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),            # Left alignment for all cells
  ]))

  # Draw table on canvas
  table.wrapOn(canvas, 0, 0)
  table.drawOn(canvas, 20, letter[1] - header_height*25)

  # Draw horizontal line
  canvas.line(20, letter[1] - header_height*25 - 20, letter[0] - 20, letter[1] - header_height*25 - 20)
  formatted_time_max = convert_to_local(max_data[0]["timestamp"])
  formatted_time_min = convert_to_local(min_data[0]["timestamp"])

  # Set font and font size for heading
  canvas.setFont("Helvetica-Bold", 11)

  # Define coordinates for the heading
  heading_x, heading_y = 40, letter[1] - header_height*35 - 40
  heading_width, heading_height = 180, 15
  canvas.setStrokeColor(colors.HexColor('#545353'))
  # Draw lines around the heading
  line_y = heading_y + heading_height / 2
  canvas.line(20, heading_y + 6, 50, heading_y + 6)
  canvas.line(210, heading_y + 6, 240,heading_y + 6)
  # Draw the heading text in the center
  heading_text = "AVERAGE MEASUREMENTS"
  text_width = canvas.stringWidth(heading_text, "Helvetica-Bold", 11)
  text_x = heading_x + (heading_width - text_width) / 2
  text_y = heading_y + (heading_height - canvas._leading) / 2
  canvas.drawString(text_x, text_y, heading_text)

  # Define heart rate data
  heart_rate_data = [
      ["Max:", f"{max_hr} ({formatted_time_max})"],
      ["Min:", f"{min_hr} ({formatted_time_min})"],
  ]
  # Create and draw the heart rate table
  heart_rate_table = Table(heart_rate_data, colWidths=[50, 90])
  heart_rate_table.setStyle(TableStyle([
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Left alignment for all cells
      ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header row text color
  ]))
  heart_rate_table.wrapOn(canvas, 0, 0)
  heart_rate_table.drawOn(canvas, 20, letter[1] - header_height * 40 + 10)

  #Measurements
  canvas.setFont("Helvetica-Bold", 11)
  # Define coordinates for the heading
  heading_x, heading_y = 40, letter[0] - 10
  heading_width, heading_height = 180, 15
  canvas.setStrokeColor(colors.HexColor('#545353'))
  # Draw lines around the heading
  line_y = heading_y + heading_height / 2
  canvas.line(20, letter[0] - 6, 60, letter[0] - 6)
  canvas.line(200, letter[0] - 6, 240, letter[0] - 6)
  # Draw the heading text in the center
  heading_text = "HEART RATE SUMMARY"
  text_width = canvas.stringWidth(heading_text, "Helvetica-Bold", 11)
  text_x = heading_x + (heading_width - text_width) / 2
  text_y = heading_y + (heading_height - canvas._leading) / 2
  canvas.drawString(text_x, text_y, heading_text)

  measurements_data = [
          ["Heart Rate:",f"{overall_mean_heartbeats} bpm"],
          ["RR Interval:",f"{overall_mean_rR} ms"],
          ["PP Interval:",f"{overall_mean_pP} ms"],
          ["PR Interval:",f"{overall_mean_pR} ms"],
          ["PR Interval:",f"{overall_mean_pR} ms"],
          # ["QS Interval:",f"{overall_mean_qS} ms"],
          # ["RT Interval:",f"{overall_mean_rT} ms"],
          ["PQ Interval:",f"{overall_mean_pQ} ms"],
          # ["RS Interval:",f"{overall_mean_rS} ms"],
          ["ST Interval:",f"{overall_mean_sT} ms"],
          # ["QR Interval:",f"{overall_mean_qR} ms"],
          ["QT Interval:",f"{overall_mean_qT} ms"],
          ["PR Interval:",f"{overall_mean_pR} ms"],
          ["QRS Interval:",f"{overall_mean_qrs} ms"],
          ["QTc Interval:",f"{overall_mean_qtc} ms"],

      ]

  # Create and draw the first heart rate table
  measurements_table = Table(measurements_data,colWidths=[80,90])
  measurements_table.setStyle(TableStyle([
          ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # Left alignment for the rest of the cells
          ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header row text color
          ('FONTSIZE', (0, 0), (-1, 0), 10),
      ]))
  measurements_table.wrapOn(canvas, 0, 0)
  measurements_table.drawOn(canvas, 20, letter[1] - header_height*65-60)

  canvas.setFont("Helvetica", 10)
  box_width = 340
  box_height = 50
  headings = [f"Ventricular Tachycardia (VT) : {len(vt_data)}" if len(vt_data) > 0 else "Ventricular Tachycardia (VT) : Not found",
              f"SVT/AT : {len(svt_data)}" if len(svt_data) > 0 else "SVT/AT : Not found",
              f"Pause : {len(pauses_data)}" if len(pauses_data) > 0 else "Pause : Not found",
              f"AFib : {len(afib_data)}" if len(afib_data) > 0 else "AFib : Not found",
              "Other : "]
  x = 250
  y = letter[0] + 30

  afib_start, pvc_start, ischemia_start, vt_start, af_start, atv_start, svt_start,bigeminy_start,trigeminy_start = calculate_start(len(pauses_data),len(afib_data),len(myocardial_ischemia_data),len(pvc_data),len(vt_data),len(af_data),len(atrioventricular_data),len(svt_data),len(bigeminy_data))
  for i, heading in enumerate(headings):
      y = y-30
      canvas.setFillColor(colors.lightgrey)
      canvas.rect(x, y , box_width, -20, stroke=0, fill=1)
      # Write heading text
      canvas.setFillColorRGB(0, 0, 0)
      canvas.drawString(x + 7, y - 15, heading)
      if heading.startswith("Ventricular") and len(vt_data) > 0:
          # Draw white box if data is available
          y = get_first_image(canvas, vt_data[0], vt_start, y)
      elif heading.startswith("SVT") and len(svt_data) > 0:
          # Draw white box if data is available
          y = get_first_image(canvas, svt_data[0], svt_start, y)
      elif heading.startswith("Pause") and len(pauses_data) > 0:
          # Draw white box if data is available
          pause_data, max_rr = get_max_interval(pauses_data)
          canvas.drawString(x + 300, y - 15, "Page 3" )
          timestamp_local = convert_to_local(pause_data["timestamp"])
          canvas.drawString(x + 65, y - 15, f'Longest RR: {timestamp_local}  duration: {max_rr} s' )
          canvas.linkURL("#page=3", (x + 300, y - 15, x + 340, y + 2))
          canvas.setFillColorRGB(1, 1, 1)  # White color
          canvas.rect(x, y-20, box_width, -box_height, stroke=1, fill=1)
          imgdata = BytesIO()
          plt = generate_ecg_image(pause_data)
          plt.savefig(imgdata, format='png', bbox_inches='tight', pad_inches=0)
          imgdata.seek(0)
          img_pause = ImageReader(imgdata)
          plt.close()
          canvas.drawImage(img_pause, x, y-20, width=box_width , height=-box_height)
          y -= box_height + 1
      elif heading.startswith("AFib") and len(afib_data) > 0:
          # Draw white box if data is available
          y = get_first_image(canvas, afib_data[0], afib_start, y)
      elif heading.startswith("Other") and (len(myocardial_ischemia_data) > 0 or len(pvc_data) or len(af_data)>0  or len(atrioventricular_data) > 0 or len(bigeminy_data) > 0 or len(trigeminy_data)>0):
          # Draw white box if data is available
          canvas.setFillColorRGB(1, 1, 1)  # White color
          canvas.rect(x, y-20 , box_width, -box_height-30, stroke=1, fill=1)
          temp = y - 20
          canvas.setFillColorRGB(0, 0, 0)
          if len(pvc_data) > 0:
            temp -=12
            canvas.drawString(x + 7, temp, f"PVC detected {len (pvc_data)} times")
            canvas.drawString(x + 300, temp, f"Page {pvc_start}" )
            canvas.linkURL(f"#page={pvc_start}", (x + 300, temp, x + 340, temp + 5))
            disease_list.append("Premature ventricular contractions")
          if len(myocardial_ischemia_data) > 0 :
            temp -=12
            canvas.drawString(x + 7, temp, f"Myocardial ischemia detected {len (myocardial_ischemia_data)} times")
            canvas.drawString(x + 300, temp, f"Page {ischemia_start}" )
            canvas.linkURL(f"#page={ischemia_start}", (x + 300, temp, x + 340, temp + 5))
            disease_list.append("Myocardial ischemia")
          if len(af_data) > 0 :
            temp -=12
            canvas.drawString(x + 7, temp, f"Atrial Flutter detected {len (af_data)} times")
            canvas.drawString(x + 300, temp, f"Page {af_start}" )
            canvas.linkURL(f"#page={af_start}", (x + 300, temp, x + 340, temp + 5))
            disease_list.append("Atrial Flutter")

          if len(atrioventricular_data) > 0 :
            temp -=12
            canvas.drawString(x + 7, temp, f"Atrioventricular block detected {len (atrioventricular_data)} times")
            canvas.drawString(x + 300, temp, f"Page {atv_start}" )
            canvas.linkURL(f"#page={atv_start}", (x + 300, temp, x + 340, temp + 5))
            disease_list.append("Atrioventricular block")

          if len(bigeminy_data) > 0 :
            temp -=12
            canvas.drawString(x + 7, temp, f"Bigeminy detected {len (bigeminy_data)} times")
            canvas.drawString(x + 300, temp, f"Page {bigeminy_start}" )
            canvas.linkURL(f"#page={bigeminy_start}", (x + 300, temp, x + 340, temp + 5))

          if len(trigeminy_data) > 0 :
            temp -=12
            canvas.drawString(x + 7, temp, f"Trigeminy detected {len (trigeminy_data)} times")
            canvas.drawString(x + 300, temp, f"Page {trigeminy_start}" )
            canvas.linkURL(f"#page={trigeminy_start}", (x + 300, temp, x + 340, temp + 5))

          y = letter[0] - 5 - i * (box_height + 10)

  #Interpretation
  interpretation_string3 = ""
  # Loop through the interpretation_data list
  interpretation_string3 += ", ".join(map(str, disease_list))
  table_width_limit = 220
  # Create a paragraph style for wrapping text
  paragraph_style = ParagraphStyle(
      'WrapStyle',
      fontSize=10,
      leading=12,
      leftIndent=0,
      rightIndent=0
  )
  # Remove the trailing comma and space

  
  interpretation_para = f"Acculive ECG monitoring done. Minimum HR - {min_hr} bpm, Maximum HR - {max_hr}, Average HR - {overall_mean_heartbeats}. " + "Symptoms for "+ interpretation_string3 +" were observed." + f"Pause noted {len(pauses_data)} times." if len(pauses_data) != 1 else f"Pause noted Only {len(pauses_data)} time. "
  if sinus_bradycardia_count > 3 : interpretation_para +=" Sinus Bradycardia detected."

  interpretation_paragraph = Paragraph(interpretation_para, paragraph_style)
  # Check if the interpretation string width exceeds the limit
#  interpretation_paragraph_width = interpretation_paragraph.wrap(table_width_limit, 0)[0]
#   if interpretation_paragraph_width > table_width_limit:
#       interpretation_string = interpretation_string[:table_width_limit]

  # Create the interpretation data
  interpretation_data = [
      ["INTERPRETATION (UNCONFIRMED)"],
      [interpretation_paragraph],
  ]

  # Create and draw the interpretation table
  interpretation_table = Table(interpretation_data, colWidths=[220])
  interpretation_table.setStyle(TableStyle([
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Left alignment for all cells
      ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),  # Header row text color
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 10),
  ]))
  interpretation_table.wrapOn(canvas, table_width_limit, 0)
  interpretation_table_height = interpretation_table._height
  interpretation_table.drawOn(canvas, 18, letter[1] - header_height*90 - 20)
  canvas.rect(20, letter[1] - header_height*90 - 20, 220,interpretation_table_height, stroke=1, fill=0)

  space_between = max(y - 100, letter[1] - header_height*90) - (letter[1] - header_height*120-40)
  cmt_height = space_between - (letter[1] - header_height*120)
  # Draw a rectangle with the calculated height
  canvas.rect(20, space_between - 10, 567, - cmt_height, stroke=1, fill=0)
  canvas.setFont("Helvetica-Bold", 10)
  canvas.setFillColorRGB(0, 0, 0)  # White color
  canvas.drawString(30,space_between - 30, "PHYSICIAN COMMENTS:")
  canvas.drawString(354, letter[1] - header_height*110 - 60 , "Signature: ")
  canvas.drawString(490, letter[1] - header_height*110 - 60, "Date:")

  page_num = canvas.getPageNumber()
  add_footer(canvas,page_num)

  canvas.showPage()
  # Add a blank box with white background
  blank_box_width = (letter[0] - 40)
  blank_box_height = 160
  canvas.setFillColorRGB(1, 1, 1)  # White color
  canvas.rect(20, letter[1] - header_height*15-10, blank_box_width, -blank_box_height, stroke=1, fill=1)
  # Add data in blank box
  heart_rate_data = [
          ["Max HR",f"{max_hr} bpm"],
          ["Min HR",f"{min_hr} bpm"],
          ["Average HR",f"{overall_mean_heartbeats} bpm"],
          ["Readable Time",f"{reading_time} hours"],
      ]
  heart_rate_table = Table(heart_rate_data, colWidths=50, rowHeights=13)
  heart_rate_table.setStyle(TableStyle([
          ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
      ]))
  heart_rate_table.wrapOn(canvas, 0, 0)
  heart_rate_table.drawOn(canvas, 490, letter[1] - header_height*30-20)
  canvas.line(460, letter[1] - header_height*30-20, letter[0] -25, letter[1] - header_height*30-20)
  canvas.setFont("Helvetica", 10)
  heart_rate_data = [
          ["","AFib",f"{len(afib_data)}"],
          ["","VT/VF",f"{len(vt_data)}"],
          ["","Pause",f"{len(pauses_data)}"],
          ["","PTE","0"],
      ]
  heart_rate_table = Table(heart_rate_data, colWidths=[470,50,45], rowHeights=13)
  heart_rate_table.setStyle(TableStyle([
          ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
          (['BACKGROUND', (0, 0), (-1, -1), colors.white]),
          (['BACKGROUND', (0, 1), (-1, 1), colors.lightgrey]),
          (['BACKGROUND', (0, 2), (-1, 2), colors.white]),
          (['BACKGROUND', (0, 3), (-1, 3), colors.lightgrey]),

      ]))
  heart_rate_table.wrapOn(canvas, 0, 0)
  heart_rate_table.drawOn(canvas, 25, letter[1] - header_height*30-75)

  #add heading
  add_header(canvas, patient_name, gender, age, pid)
  canvas.setFont("Helvetica-Bold", 16)
  canvas.setFillColorRGB(0, 0, 0)  # White color
  canvas.drawString(20, letter[1] - header_height*10-10, "Day 1 Report Summary")

  x, y = 20, letter[1] - header_height*50 - 20
  width, height = 150,30
  add_heading(canvas, x, y, width, height, "Heart Rate Trend")
  imgdata = BytesIO()
  plt = generate_plot(hours, heartbeats_list)  # Adjust as needed
  plt.savefig(imgdata, format='png', bbox_inches='tight', pad_inches=0)
  imgdata.seek(0)
  img_24hr = ImageReader(imgdata)
  plt.close()
  # Reading time
  canvas.drawImage(img_24hr, 25, letter[1] - header_height*30-50, width=430, height=100)
  # Max hr image
  start_y = letter[1] - header_height*60 - 10
  start_y = add_images(canvas, max_data, start_y,"Max HR" ,1, patient_name, gender, age, pid, json_data)
  # Min hr image
  start_y = letter[1] - header_height*80-50
  start_y = add_images(canvas, min_data, start_y,"Min HR" ,2, patient_name, gender, age, pid, json_data)

  page_num = canvas.getPageNumber()
  add_footer(canvas,page_num)

  # Page 3
  canvas.showPage()
  i=3
  canvas.setLineWidth(2)
  canvas.line(20, letter[1] - header_height*15, letter[0] - 20, letter[1] - header_height*15)
  canvas.setLineWidth(1)
  start_y = letter[1] - header_height * 10 - 25

  # Pause
  if pause:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          start_y = letter[1] - header_height * 10 - 75
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Pause")
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_header(canvas, patient_name, gender, age, pid)
          add_heading(canvas, x, y, width, height, "Pause")
          start_y -=50
      disease = "Pause detected at"
      i, start_y = add_images(canvas, pauses_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  # Afib
  if afib:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          start_y = letter[1] - header_height * 10 - 75
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Atrial Fibrillation")
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Atrial Fibrillation")
          start_y -=50
      disease = "AFib detected at"
      i, start_y = add_images(canvas, afib_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30
  # PVC
  if pvc:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 300,30
          add_heading(canvas, x, y, width, height, "Premature Ventricular Contractions")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 300, 30
          add_heading(canvas, x, y, width, height, "Premature Ventricular Contractions")
          start_y -=50
      disease = "PVC detected at"
      i, start_y = add_images(canvas, pvc_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  # myocardial ischemia
  if len(myocardial_ischemia_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Myocardial ischemia")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Myocardial ischemia")
          start_y -=50
      disease = "ischemia detected at"
      i, start_y = add_images(canvas, myocardial_ischemia_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  # VT
  if len(vt_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Ventricular Tachycardia")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Ventricular Tachycardia")
          start_y -=50
      disease = "VT detected at"
      i, start_y = add_images(canvas, vt_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30
  #AtrialFlutter
  if len(af_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 150,30
          add_heading(canvas, x, y, width, height, "Atrial Flutter")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Atrial Flutter")
          start_y -=50
      disease = "AF detected at"
      i, start_y = add_images(canvas, af_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  #Atrioventricular block
  if len(atrioventricular_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 250,30
          add_heading(canvas, x, y, width, height, "Atrioventricular Block")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 250, 30
          add_heading(canvas, x, y, width, height, "Atrioventricular Block")
          start_y -=50
      disease = "ATV block detected at"
      i, start_y = add_images(canvas, atrioventricular_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  #SVT
  if len(svt_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 250,30
          add_heading(canvas, x, y, width, height, "Superventricular Tachycardia")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 250, 30
          add_heading(canvas, x, y, width, height, "Superventricular Tachycardia")
          start_y -=50
      disease = "SVT detected at"
      i, start_y = add_images(canvas, svt_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  #Bigeminy
  if len(bigeminy_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Bigeminy Signals")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Bigeminy Signals")
          start_y -=50
      disease = "Bigeminy detected at"
      i, start_y = add_images(canvas, bigeminy_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30

  #Trigeminy
  if len(trigeminy_data) > 0:
      if start_y - 200 < 8:
          page_num = canvas.getPageNumber()
          add_footer(canvas,page_num)
          canvas.showPage()
          add_header(canvas, patient_name, gender, age, pid)
          x, y = 20, letter[1] - header_height*10 - 25
          width, height = 200,30
          add_heading(canvas, x, y, width, height, "Trigeminy Signals")
          start_y = letter[1] - header_height * 10 - 75
      else:
          canvas.setFillColorRGB(0, 0, 0)  # Black color
          canvas.setFont("Helvetica-Bold", 16)
          x, y = 20, start_y
          width, height = 200, 30
          add_heading(canvas, x, y, width, height, "Trigeminy Signals")
          start_y -=50
      disease = "Trigeminy detected at"
      i, start_y = add_images(canvas, trigeminy_data, start_y,disease ,i, patient_name, gender, age, pid, json_data)
      start_y -=30
  page_num = canvas.getPageNumber()
  add_footer(canvas,page_num)
  # Save the PDF
  canvas.save()
  pdf_buffer.seek(0)
  return pdf_buffer
