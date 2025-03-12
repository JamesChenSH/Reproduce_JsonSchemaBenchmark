# This file contains the Logger class which is used to log messages to the console.
class Logger:
    
    verbose: bool
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def log(self, message: str, force: bool = False, header=None):
        header = "[LOG]" if header is None else header
        if force or self.verbose:
            print(f"{header}: {message}", flush=True)
            