import logging
from configs import runInfo

def make_logger(logfile, name = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(levelname)s|%(asctime)s] - %(filename)s(%(lineno)s) - %(message)s")
    console = logging.StreamHandler()
    
    file_handler = logging.FileHandler(filename=logfile)
    console.setLevel(runInfo.console_log_level)
    file_handler.setLevel(runInfo.file_log_level)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger