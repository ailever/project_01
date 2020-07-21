import pprint
import sys, os

class trace:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        stdout_restore = sys.stdout                 # Save the current stdout so that we can revert sys.stdou after we complete
        sys.stdout = open('temp', 'w')           # Redirect sys.stdout to the file
        print(self.func(*args, **kwargs))
        sys.stdout.close()			        # Close the file
        sys.stdout = stdout_restore		        # Restore sys.stdout to our old saved file handler
        
        with open('temp', 'r') as f:
            lines = f.readlines()
        os.remove('temp')
        
        stdout = []
        for line in lines:
            stdout.append(line[:-1])

        return self.func(*args, **kwargs), stdout


@trace
def _getattr(obj, i):
    return type(getattr(obj, i))

def information(obj):
    print(f'\n* vars(obj)')
    for i in vars(obj):
        print(f' - obj.{i} : {getattr(obj, i)}')
        _, stdout = _getattr(obj, i); stdout = stdout[0][8:-2]
        if stdout == 'collections.OrderedDict':
            for key in getattr(obj, i):
                print(f'  - obj.{i}["{key}"] : {getattr(obj, i)[key]}')


    print(f'\n* dir(obj)')
    for i in vars(obj):
        print(f' - obj.{i} : {getattr(obj, i)}')

