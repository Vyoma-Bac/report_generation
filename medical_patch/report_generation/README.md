# ECG Report Generation
## Project setup:
1. Create venv file
    - `pip install virtualenv`
    - `python -m venv venv`
2. Activate virtualenv
    - For windows:
      - `venv\Scripts\activate`
    - For Ubuntu:
      - `source venv/bin/activate`
3. Install requirement dependencies 
    - `pip install -r requirements.txt`

## Steps to run project:
1. Run python file
   - `python app.py`
2. Open following link in browser
   - `http://127.0.0.1:5000/`

## Procedure to generate report :
1. send request on following endpoint:
   http://127.0.0.1:5000/download-report?userId={userId}&startDate={startDate}&endDate={endDate}
  
