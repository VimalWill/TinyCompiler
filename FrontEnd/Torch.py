try:
    from torch_mlir.torchscript import OutputType, compile
except ImportError:
    raise ImportError("failed to import 'torch_mlir'")
import torch

def emitMLIR(Module : torch.nn.Module, Tensor : torch.Tensor):
    """
    emits MLIR for the passed torch module

    parameters:
        @Module - PyTorch module / model
        @Tensor - Sample InputTensor
        @return - MLIR version of torch module
                  esp. in Tosa dialect
    """
    return compile(Module, Tensor, OutputType.TOSA)
