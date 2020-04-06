# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file

import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

original_stdout = None


def stdout_add_file(file):
    f = open(file, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)


def stdout_reset():
    if original_stdout is not None:
        sys.stdout = original_stdout