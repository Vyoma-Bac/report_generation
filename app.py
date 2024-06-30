from flask import Flask, make_response, request, jsonify
from utils.utils import generate_pdf, fetch_user_data
import time
import requests
import threading
import pikepdf
import io
############change complaints portion
app = Flask(__name__)

def compress_pdf_with_pikepdf(input_pdf_buffer):
    # Open the input PDF from the buffer
    input_pdf_buffer.seek(0)
    pdf_document = pikepdf.open(input_pdf_buffer)

    # Create a BytesIO buffer for the compressed PDF
    compressed_pdf_buffer = io.BytesIO()

    # Save the PDF with optimization
    pdf_document.save(compressed_pdf_buffer)
    pdf_document.close()

    # Return the compressed PDF buffer
    compressed_pdf_buffer.seek(0)
    return compressed_pdf_buffer

def send_pdf_to_api(user_id, start_date, end_date, doctor_email, customer_name):
    try:
        # Generate PDF
        pdf_buffer, formatted_date_record_for  = generate_pdf(user_id, start_date, end_date)
        
        # Check the size of the PDF
        pdf_buffer.seek(0, io.SEEK_END)
        pdf_size = pdf_buffer.tell()
        pdf_buffer.seek(0)
        print(f"Original PDF size: {pdf_size} bytes")

        # Compress the PDF if it is larger than 15 MB (15 * 1024 * 1024 bytes)
        if pdf_size > 15 * 1024 * 1024:
            print("PDF is larger than 15 MB, compressing...")
            compressed_pdf_buffer = compress_pdf_with_pikepdf(pdf_buffer)
        else:
            print("PDF is less than 15 MB, no compression needed.")
            compressed_pdf_buffer = pdf_buffer
        # Replace with https://api.accu.live
        # Send PDF to API
        doctor_email = 'vyoma.suthar@bacancy.com'

        target_api_url = 'https://bac-accu-live-1-0-0.onrender.com/api/v1/send-report/'
        files = {'report': ('ecg_report.pdf', compressed_pdf_buffer, 'application/pdf')}
        data = {
            'doctorEmail': doctor_email,
            'customerName': customer_name,
            'date': formatted_date_record_for
        }
        print("Sending PDF to API...")
        api_response = requests.post(target_api_url, files=files, data=data)
        api_response.raise_for_status()  # This will raise an HTTPError for bad responses
        print('PDF sent successfully to the target API')
    except requests.exceptions.RequestException as e:
        print(f'Failed to send PDF to the target API: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

@app.route('/download-report', methods=['GET'])
def download_report():
    start_time = time.time()
    print("Request received at:", time.ctime(start_time))

    userId = request.args.get('userId')
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')

    # Fetch user data
    user_data = fetch_user_data(userId)
    doctor_email = user_data['doctor_email']
    customer_name = user_data['patient_name']
    if not user_data:
        print("No data found for the provided user ID")
        return jsonify({"error": "No data found for the provided user ID"}), 404

    # Send the preliminary response immediately
    response_data = {"message": "Preparing to generate report and send to your email. Please wait..."}
    response = jsonify(response_data)
    # Start a new thread to generate PDF and send to API
    threading.Thread(target=send_pdf_to_api, args=(userId, startDate, endDate, doctor_email, customer_name)).start()

    # Return the preliminary response immediately
    return response

# @app.route('/download-report', methods=['GET'])
# def download_report():
#     start_time = time.time()
#     print("start_time", start_time)

#     userId = request.args.get('userId')
#     startDate = request.args.get('startDate')
#     endDate = request.args.get('endDate')

#     pdf_buffer = generate_pdf(userId, startDate, endDate)
#     #compressed_pdf_buffer = compress_pdf_with_pikepdf(pdf_buffer)

#     # Prepare Flask response
#     response = make_response(pdf_buffer)
#     response.headers['Content-Type'] = 'application/pdf'
#     response.headers['Content-Disposition'] = 'attachment; filename=ecg_report.pdf'

#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Download report request took {duration:.4f} seconds")

#     return response


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
