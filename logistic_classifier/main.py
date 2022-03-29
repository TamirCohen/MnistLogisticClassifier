import numpy
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('application logger')
logger.setLevel(logging.INFO)
"""
CONFIG
MATCH CASE
PROFILE
"""
if __name__ == "__main__":
   logger.info("test")
