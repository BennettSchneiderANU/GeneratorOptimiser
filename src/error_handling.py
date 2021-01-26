import os
import sys
import traceback
import inspect
import logging

def strConList(myList):
    """
    This functions takes a list and concatenates the elements with a newline character.
    This function is usefull for printing
    
    input:
        myList (list): the list you want to concatenate     
    output:
        myList_str (str): A string of all the elements in myList seperated by newLine characters
        
           
    Created on 13/11/2020 by Hilbert van Pelt, code by Bennett Schneider 
    
    """

    
    myList_str = '\n'.join([str(m) for m in myList]) # modify myList so each element prints as a new line

    return myList_str


def errFunc(message,handled=False,ext=True,myList=[]):
    """
    This function returns a nicely-formatted and informative error message.
    It should only be called within an exception. 

    Args:
        message (str): If this string is likely to cut across multiple lines,
            it should be formatted like this: '("line1" + " line2" + ... + " lineN")'.
            This message should tell the user what has likely caused the error, and 
            what can be done to fix or troubleshoot it.
        
        handled (bool. Default = False): This controls whether the traceback error
            message is printed.
            True -> traceback error message not printed (exception in handled)
            False -> traceback error message is printed (exception not handled) (default)

        ext (bool. Default = True): This controls whether the function induces
            a system exit due to the recent error caught by the exception:
            True -> System will exit (default)
            False -> System will not exit
            Note that if 'handled' = True, this value will automatically be set to False, 
            i.e. the system will not exit, regardless of the original setting. 

        myList (list-like. Default=[]): An iterable variable that will be printed out in full
            with each element on a new line. It will print after the message with a line
            break in between. Make sure the message introduces the variable being printed.
    
    Functions used:
        - misc_functions:
            - caller_name

    Use this function within an exception field to return a clear error message and 
    relevant error information to the user.

    Created on 07/02/2019 by Bennett Schneider

    """
    callerName = caller_name(skip=2) # get the package, class, and function name

    
    myList_str = '\n'.join([str(m) for m in myList]) # modify myList so each element prints as a new line

    logging.debug("_____________________________________________________\n\n")

    if not handled: # if the error was not handled       
        
        logging.error(f"Error raised in {callerName}") 

        logging.debug(f":\n{traceback.format_exc()}\n") # traceback error message

    
    else: # if the error was handled
        logging.info(f"Error handled in {callerName}:\n") 

        ext = False # never allow system to exit from a handled error
        
    logging.warning(f"{message}\n") 
    # Print out list-like objects in a nice way, outside the message string
    if myList != []:
        logging.warning(myList_str)

    logging.debug("_____________________________________________________")

    if ext ==  True: # if we want to exit the program
        logging.critical('\nExiting program - Try again!')
        sys.exit() # best way to induce a system exit
        raise SystemExit


def caller_name(skip=2):
    """Get a name of a caller in the format module.class.method
    
       `skip` specifies how many levels of stack to skip while getting caller
       name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.
       
       An empty string is returned if skipped levels exceed stack height

       Created in 2012 by 'techtonik'.
       Original code can be found here: https://gist.github.com/techtonik/2151727
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
      return ''
    parentframe = stack[start][0]    
    
    name = []
    module = inspect.getmodule(parentframe)
    # `modname` can be None when frame is executed directly in console
    # TODO(techtonik): consider using __main__
    if module:
        name.append(module.__name__)
    # detect classname
    if 'self' in parentframe.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':  # top level usually
        name.append( codename ) # function or a method
    del parentframe
    return ".".join(name)

def SetLogging(path = '', level = 'INFO'):
    """
    
    Use this function to set the logging
    Inputs:
        path (string): The path to the file were you want the log to reside
        
        level (string): The log level you want
    
    """
    #set log level
    if level == 'DEBUG':
        logLevel = logging.DEBUG
        
    elif level == 'INFO':
        logLevel = logging.INFO
        
    elif level == 'WARNING':
        logLevel = logging.WARNING
        
    elif level == 'ERROR':
        logLevel = logging.ERROR
        
    elif level == 'CRITICAL':
        logLevel = logging.CRITICAL
    else:
        sys.exit('loglevel options are: DEBUG, INFO, WARNING, ERROR, CRITICAL. Please select one of those')
    
    # if no path is set run this format
    if path == '':
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig( level=logLevel,
                           filemode='w',
                           format='%(levelname)s:%(message)s')
                           #format = '%(levelname)s:%(filename)s:%(lineno)s %(levelname)s:%(message)s')
                           
    #With path run the following
    else:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename = path, 
                            level=logLevel,
                            filemode='w',
                            #format='%(levelname)s:%(message)s')
                            format = '%(levelname)s:%(filename)s:%(lineno)s:%(message)s')
        
        
    logging.info('Starting logging with level '+level)