import torch
from typing import Annotated
def eltwise_add(x: Annotated[torch.Tensor, "shape: (1024)"], y: Annotated[torch.Tensor, "shape: (1024)"]) -> Annotated[torch.Tensor, "shape: (1024)"]:
    """
    x: first input vector tensor
    y: second input vector tensor
    """
    return x + y