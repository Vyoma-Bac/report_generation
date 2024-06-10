
from flask import Flask, make_response, request
from utils import utils

app = Flask(__name__)

@app.route('/download-report', methods=['GET'])
def download_report():
    userId = request.args.get('userId')
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')
    print("yesss")
    pdf_buffer = utils.generate_pdf(userId, startDate, endDate)
    response = make_response(pdf_buffer)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=ecg_report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)
