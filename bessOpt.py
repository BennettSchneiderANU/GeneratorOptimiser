import os
import sys

# set path to the current file
myPath = os.path.dirname(os.path.realpath(__file__))

# add myPath to sys.path so the other modules can find it
if myPath not in sys.path:
    sys.path.append(myPath)

