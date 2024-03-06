import inspect

_verbose = 1


def set_log_level(verbose):
    global _verbose
    _verbose = verbose


def get_log_level():
    return _verbose


def debug_info(message:str):
    caller_frame = inspect.stack()[1]
    caller_filename = caller_frame[1]
    caller_lineno = caller_frame[2]
    print(f"({caller_filename}:{caller_lineno}):\n{message}")