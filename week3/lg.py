import logging

logging.basicConfig(filename='app.log', filemode='a',format='%(asctime)s - %(message)s')
name = 'John'

logging.error('%s raised an error', name)
logging.warning('This will get logged to a file')
a = 5
b = 0

try:
  c = a / b
except Exception as e:
  logging.error("Exception occurred", exc_info=True)
logger = logging.getLogger('example_logger')
logger.warning('This is a warning')