from flask import Flask, make_response, request, jsonify
from utils.utils import generate_pdf, fetch_user_data
import time
import requests
import threading
import pikepdf
import io

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

def send_pdf_to_api(user_id, start_date, end_date):
    try:
        # Generate PDF
        pdf_buffer = generate_pdf(user_id, start_date, end_date)
        
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

        # Send PDF to API
        target_api_url = 'https://bac-accu-live-1-0-0.onrender.com/api/v1/send-report/'
        files = {'report': ('ecg_report.pdf', compressed_pdf_buffer, 'application/pdf')}
        data = {
            'doctorEmail': 'vyoma.suthar@bacancy.com',
            'customerName': 'Test',
            'date': '28-06-2024'
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
    if not user_data:
        print("No data found for the provided user ID")
        return jsonify({"error": "No data found for the provided user ID"}), 404

    # Send the preliminary response immediately
    response_data = {"message": "Preparing to generate report and send to your email. Please wait..."}
    response = jsonify(response_data)

    # Start a new thread to generate PDF and send to API
    threading.Thread(target=send_pdf_to_api, args=(userId, startDate, endDate)).start()

    # Return the preliminary response immediately
    return response

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
