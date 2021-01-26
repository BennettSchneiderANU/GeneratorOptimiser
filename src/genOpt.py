import os
import sys
import traceback
import inspect
import logging

# set path to the current file
myPath = os.path.dirname(os.path.realpath(__file__))

# add myPath to sys.path so the other modules can find it
if myPath not in sys.path:
    sys.path.append(myPath)

from classes import *
from plotting_functions import *
from analysis_functions import *
from error_handling import *

# Set the logging package
SetLogging()