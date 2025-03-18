# This file contains the Logger class which is used to log messages to the console.
class Logger:
    
    verbose: bool
    
    def __init__(self, verbose: bool = False, file_path: str = None):
        self.verbose = verbose
        self.file_path = file_path
        
    def log(self, message: str, force: bool = False, header=None, to_file=False):
        header = "[LOG]" if header is None else header
        if force or self.verbose:
            print(f"{header}: {message}", flush=True)
        if to_file:
            with open(self.file_path, 'a') as f:
                f.write(f"{header}: {message}\n")
            