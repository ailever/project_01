import sys, os
from collections import OrderedDict


class Attribute:
    def rc_get(self, obj, memory='', count=0):
        if hasattr(obj, '__dict__') and len(obj.__dict__) != 0:
            self._rc_get_obj(obj, memory, count)
        elif isinstance(obj, list):
            self._rc_get_list(obj, memory, count)
        elif isinstance(obj, dict):
            self._rc_get_dict(obj, memory, count)

    def _rc_get_obj(self, obj, memory='', count=0):
        if hasattr(obj, '__dict__') and len(obj.__dict__) != 0:
            for key, attr in vars(obj).items():
                if not len(memory):
                    print(f'.{key} : {attr}')
                else:
                    print(f'{memory}.{key} : {attr}')

            self.spliter(n=count)
            for key, attr in vars(obj).items():
                if hasattr(attr, '__dict__') and len(obj.__dict__) != 0:
                    self._rc_get_obj(attr, memory=memory+'.'+str(key), count=count+1)
                else:
                    self.rc_get(attr, memory=memory+'.'+str(key), count=count)
        else:
            return None

    def _rc_get_list(self, obj, memory='', count=0):
        if isinstance(obj, list):
            for key, attr in enumerate(obj):
                if not len(memory):
                    print(f'[{key}] : {attr}')
                else:
                    print(f'{memory}[{key}] : {attr}')
            
            self.spliter(n=count)
            for key, attr in enumerate(obj):
                if isinstance(attr, list):
                    self._rc_get_list(attr, memory=memory+'['+str(key)+']', count=count+1)
                else:
                    self.rc_get(attr, memory=memory+'['+str(key)+']', count=count)
        
        else:
            return None

    def _rc_get_dict(self, obj, memory='', count=0):
        if isinstance(obj, (dict, OrderedDict)):
            for key, attr in obj.items():
                if not len(memory):
                    if isinstance(key, str):
                        print(f'["{key}"] : {attr}')
                    else:
                        print(f'[{key}] : {attr}')
                else:
                    if isinstance(key, str):
                        print(f'{memory}["{key}"] : {attr}')
                    else:    
                        print(f'{memory}[{key}] : {attr}')
            
            self.spliter(n=count)
            for key, attr in obj.items():
                if isinstance(attr, (dict, OrderedDict)):
                    if isinstance(key, str):
                        self._rc_get_dict(attr, memory=memory+'["'+str(key)+'"]', count=count+1)
                    else:
                        self._rc_get_dict(attr, memory=memory+'['+str(key)+']', count=count+1)
                else:
                    if isinstance(key, str):
                        self.rc_get(attr, memory=memory+'["'+str(key)+'"]', count=count)
                    else:
                        self.rc_get(attr, memory=memory+'['+str(key)+']', count=count)
        else:
            return None

    @staticmethod
    def spliter(n, turn=False):
        split_line = '-'*(100 - 5*n)
        if turn==True:
            print(split_line)



class logtrace:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        if not os.path.isdir('DebuggingLog'):
            os.mkdir('DebuggingLog')
            num = 0
        elif not os.listdir('DebuggingLog/'):
            num = 0
        else:
            loglist = os.listdir('DebuggingLog/')
            
            lognumbers = []
            for log in loglist:
                lognumbers.append(int(log[13:]))
            num = max(lognumbers) + 1

        stdout_restore = sys.stdout                                         # Save the current stdout so that we can revert sys.stdou after we complete
        sys.stdout = open(f'DebuggingLog/debugging.log{num}', 'w')          # Redirect sys.stdout to the file
        
        logs = kwargs['logs']           # logs : self.logs in Debugger
        lognames = kwargs['lognames']     # lognames : self.namespace in Debugger
        
        print('* FILE NAME :', sys.argv[0])
        print('* BREAK POINT', set(logs.keys()))
        for key in logs:
            obj = logs[key]
            print(f'  * {key} : 0~{len(obj)-1}')
        
        print('\n* -----------------------------------------------DETAILS INFO(attributes)-------------------------------------------*')
        attribute = Attribute()

        for key, values in logs.items():
            for i, obj in enumerate(values):
                print()
                print(f'[{key}][{i}] - {i}th object')
                print(f'===========================')
                print(attribute.rc_get(obj))

        sys.stdout.close()              # Close the file
        sys.stdout = stdout_restore     # Restore sys.stdout to our old saved file handler      
        print(f'DebuggingLog/debugging.log{num} file was sucessfully created!')
        return self.func(*args, **kwargs)



class Debugger:
    def __init__(self):
        self.attribute = list()
        self.forlooplog = list()
        self.namespace = list()
        self.logs = OrderedDict()
        self.callcount = -1

    def __call__(self, *obj, **kwargs):
        self.callcount += 1
        self.logs[kwargs['logname']] = obj

        self.forlooplog.append(obj)
        if 'logname' in kwargs:
            self.namespace.append(f'[debug{self.callcount}] '+kwargs['logname'])
        else:
            self.namespace.append(f'[debug{self.callcount}] Untitled')

    def __del__(self):
        for name in set(self.namespace):
            if name[-8:] != 'Untitled':
                pass
        self.logwriter(self.forlooplog, lognames=self.namespace, logs=self.logs)

    @logtrace
    def logwriter(self, *args, **kwargs):
        pass

