# -*- coding: utf-8 -*-
"""
Created on 11 June 2023
@author: jemccarthy13

Creates a custom logging utility.
Use:
import logging
import psLog from PSLogger
psLog.setLevel(logging.xxx)
psLog.debug(...)
psLog.info(...)
psLog.warning(...)
psLog.error(...)
psLog.critical(...)

Logging level of ERROR will only display critical|error.
Logging level of WARNING will only display warning|critical|error.
Logging level of INFO will display crit & error & warning & info.
Logging level of DEBUG displays all.
"""
import logging

# Create the ParrotSour logger / set initial level to DEBUG
psLog = logging.getLogger('PS')
psLog.setLevel(logging.DEBUG)

# Save time by not capturing logging metadata
logging.logThreads = False
logging.logProcesses = False
logging.logMultiprocessing = False
logging._srcfile = None

# create console handler and set level to debug
# note - do not change this value. Instaed,
# psLog.setLevel(logging.XXX) to change the level of logging
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('>> %(levelname)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
psLog.addHandler(ch)
