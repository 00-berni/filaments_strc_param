from .data import * 
from .display import * 
from .stuff import *
from .fit_stuff import get_data_fit
import logging

filpy_logger = logging.getLogger(__name__)
filpy_logger.info('Prova')
INITIAL_STR = f"Package Dir: {PKG_DIR}\nProject Dir: {PROJECT_DIR}"

print(INITIAL_STR)