import torch
import torch_mlir.fx as fx
from torch_mlir.compiler_utils import OutputType

def tinycompile(fn):
    '''
    torch_mlir-based front-end to emit
    TOSA dialect to support AOT compilation 
    in Tiny-Compiler. 

    return:
        mlir-ir : str
    '''
    def wrapper():
        Module: torch.nn.Module  = None
        Data: torch.Tensor = None
        
        Module, Data = fn()
        func_name = Module.__class__.__name__

        mlir_rep = fx.export_and_import(
            Module, 
            Data, 
            output_type = OutputType.TOSA, 
            func_name = func_name
        )
        return mlir_rep
    
    return wrapper


