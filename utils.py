import os
import sys


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = old_stdout
        # pass the return value of the method back
        return value

    return func_wrapper



