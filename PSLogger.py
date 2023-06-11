import logging

psLog = logging.getLogger('PS')
psLog.setLevel(logging.DEBUG)

logging.logThreads = False
logging.logProcesses = False
logging.logMultiprocessing = False
logging._srcfile = None

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('>> %(levelname)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
psLog.addHandler(ch)
