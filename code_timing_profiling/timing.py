from functools import wraps
import time
import numpy as np


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Time execution of module {} function {} is {} seconds".format(func.__module__, func.__name__, end - start))
        return result
    return wrapper
