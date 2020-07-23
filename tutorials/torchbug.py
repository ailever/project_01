import os, sys
from collections import OrderedDict
import torch
import re

class logtrace:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        if not os.path.isdir('.DebuggingLog'):
            os.mkdir('.DebuggingLog')
            num = 0
        elif not os.listdir('.DebuggingLog/'):
            num = 0
        else:
            loglist = os.listdir('.DebuggingLog/')
            
            lognumbers = []
            for log in loglist:
                if re.search(r'^debugging\.log', log):
                    lognumbers.append(int(log[13:]))
            if len(lognumbers) == 0:
                num = 0
            else:
                num = max(lognumbers) + 1

        stdout_restore = sys.stdout                                         # Save the current stdout so that we can revert sys.stdou after we complete
        sys.stdout = open(f'.DebuggingLog/debugging.log{num}', 'w')          # Redirect sys.stdout to the file
        """
        file info overview!
        """
        forlooplog = kwargs['forlooplog']
        logs = kwargs['logs']           # logs : self.logs in Debugger
        lognames = kwargs['lognames']     # lognames : self.namespace in Debugger
        
        print('* FILE NAME :', sys.argv[0])
        print('* BREAK POINT', set(logs.keys()))
        for key in logs:
            values = logs[key]
            print(f'  * {key} : 0~{len(values)-1}')
            
        for key, values in logs.items():
            for i, obj in enumerate(values):
                print(f'\n[{key}][{i}] - {i}th object')
                print(f'===========================')
                if isinstance(obj, torch.Tensor):
                    print('tensor size()              : ', obj.size())
                    print('tensor type()              : ', obj.type())


        
        print('\n* [1]-----------------------------------------------DETAILS INFO(attributes)-------------------------------------------*')
        
        """
        write, here!
        ['requires_grad', 'is_leaf', 'retain_grad', 'grad_fn', 'grad']
        """
        for key, values in logs.items():
            for i, obj in enumerate(values):
                print(f'\n[{key}][{i}] - {i}th object')
                print(f'===========================')
                if isinstance(obj, torch.Tensor):
                    print('tensor dim()               : ', obj.dim())
                    print('tensor size()              : ', obj.size())
                    print('tensor numel()             : ', obj.numel())
                    print('tensor type()              : ', obj.type())
                    print('tensor is_signed()         : ', obj.is_signed())
                    print('tensor is_complex()        : ', obj.is_complex())
                    print('tensor is_floating_point() : ', obj.is_floating_point())
                    print('tensor is_cuda             : ', obj.is_cuda)
                    print('tensor is_shared()         : ', obj.is_shared())
                    print('tensor device              : ', obj.device)
                    print('tensor is_quantized        : ', obj.is_quantized)
                    print('tensor is_contiguous()     : ', obj.is_contiguous())
                    print('tensor is_pinned()         : ', obj.is_pinned())
                    print('tensor requires_grad       : ', obj.requires_grad)
                    print('tensor is_leaf             : ', obj.is_leaf)
                    print('tensor grad_fn             : ', obj.grad_fn)
                    print('tensor data                :\n', obj.data)
                    print('tensor grad                :\n', obj.grad)
                    print('tensor retain_grad         :\n', obj.retain_grad)

        sys.stdout.close()              # Close the file
        sys.stdout = stdout_restore     # Restore sys.stdout to our old saved file handler      
        print(f'.DebuggingLog/debugging.log{num} file was sucessfully created!')
        return self.func(*args, **kwargs)


class Debugger:
    def __init__(self):
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
        self.logwriter(None,
                       forlooplog=self.forlooplog,
                       lognames=self.namespace,
                       logs=self.logs)
    
    @logtrace
    def logwriter(self, *args, **kwargs):
        pass


def main():
    x = torch.Tensor(3,3).uniform_(0,1)
    x.requires_grad_()
    y = 2*x
    z = y**2
    loss = z.sum()
    loss.backward()

    debugger = Debugger()
    debugger(x, y, z, loss, logname='here')
    del debugger


if __name__ == "__main__":
    main()
