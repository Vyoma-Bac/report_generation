
from flask import Flask, make_response, request
from utils.utils import generate_pdf
import time
import asyncio

app = Flask(__name__)

@app.route('/download-report', methods=['GET'])
def download_report():
    start_time = time.time()
    print("start_time",start_time)
    userId = request.args.get('userId')
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')
    pdf_buffer = asyncio.run(generate_pdf(userId, startDate, endDate))
    response = make_response(pdf_buffer)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=ecg_report.pdf'
    end_time = time.time()
    duration = end_time - start_time
    print(f"Download report request took {duration:.4f} seconds")
    return response

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)