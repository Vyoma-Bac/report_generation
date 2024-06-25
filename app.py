
from flask import Flask, make_response, request, jsonify
from utils.utils import generate_pdf
# from utils.test import fetch_data
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

# @app.route('/test_data', methods=['GET'])
# def test_data():
#     start_time = time.time()
#     print("start_time", start_time)
    
#     userId = request.args.get('userId')
#     startDate = int(request.args.get('startDate'))  # Convert to integer if needed
#     endDate = int(request.args.get('endDate'))      # Convert to integer if needed
    
#     # Run fetch_ecg_data_parallel asynchronously
#     # async def fetch_data():
#     #     return await test.fetch_ecg_data_parallel(userId, startDate, endDate)
    
#     # # Execute asynchronous function using asyncio.run
#     # try:
#     #         loop = asyncio.get_running_loop()
#     # except RuntimeError:
#     #         loop = asyncio.new_event_loop()
#     #         asyncio.set_event_loop(loop)
    
#     # Execute asynchronous function using asyncio.run
    
#     ecg_data = fetch_data(userId, startDate, endDate)
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"Download report request took {duration:.4f} seconds")
#     return jsonify(ecg_data)


if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)
