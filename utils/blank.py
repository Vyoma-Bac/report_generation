from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
import os
from datetime import datetime, timezone
import pytz

def generate_blank_pdf(patient_name, ref_doc, age, gender, pid, mobile_no, startDate, endDate):
    ist = pytz.timezone('Asia/Kolkata')
    start_date_ist = datetime.fromtimestamp(startDate / 1000, tz=timezone.utc).astimezone(ist).strftime('%Y-%m-%d %H:%M:%S')
    end_date_ist = datetime.fromtimestamp(endDate / 1000, tz=timezone.utc).astimezone(ist).strftime('%Y-%m-%d %H:%M:%S')
    
    header_height = 6
    logo = os.path.join('static/images', 'Full Logo - On Light.png')
    pdf_buffer_first_page = BytesIO()
    canvas = Canvas(pdf_buffer_first_page, pagesize=letter)
    canvas.setPageCompression(True)
    canvas.drawImage(logo, 20, letter[1] - 60, width=150, height=40)
    # # Add heading
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(400, letter[1] - 55, "ECG REPORT")
    canvas.line(20, letter[1] - header_height*10-10, letter[0] - 20, letter[1] - header_height*10-10)
    data = [
        [f"Name: {patient_name}","","","","","",f"Referal Doctor: {ref_doc}"],
        [f"Age: {age} Yr        Gender: {gender}","","","","","",f"ID: {pid}","",f"Tel: {int(mobile_no)}"],
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
    # Add smaller text below the table
    canvas.setFont("Helvetica", 12)
    text_y_position = letter[1] - header_height*25 - (len(data) * 14) - 20  # Adjust the position based on the table height
    canvas.drawString(20, text_y_position, f"No data found from {start_date_ist} to {end_date_ist}")
    canvas.save()
    pdf_buffer_first_page.seek(0)
    
    return pdf_buffer_first_page, start_date_ist