################################################################################
# CleanSomas — zero-output soma block for PerforatedAI dendrite experiments.  #
################################################################################

#
"""
Imports
"""
import torch

from torch import Tensor, nn
from typing import Any, Union


#
"""
Model
"""
class CleanSomas(nn.Module):
    '''
    Soma block that contributes no direct input signal to the dendritic ANN

    Notes:
        - With bias=True (default): returns a learned bias vector broadcast
          over the batch, exactly equivalent to a MaskedLinear soma whose
          receptive-field mask is all zeros.  The bias is initialised to zeros
          matching MaskedLinear.reset_parameters behaviour.
        - With bias=False: returns a zero tensor, no learnable parameters.
        - PAI wraps this module and adds dendrite outputs on top of whatever
          this returns, so the dendrites carry the full input signal either way.
        - config is stored as-is and is accessible to any dendrite factory
          that receives this module as parent_module.

    Signature:
        output_dims (int | tuple):
            - Number of output units, or a tuple of output dimensions.
              The batch axis is prepended automatically in forward.
        config (dict):
            - Arbitrary metadata; stored verbatim, never inspected here.
        bias (bool):
            - True  → learnable bias, functionally identical to a zero-rf soma.
            - False → pure zeros, no parameters.
    '''

    def __init__(
        self,
        output_dims: Union[int, tuple],
        config     : dict,
        bias       : bool = True,
    ) -> None:
        super().__init__()
        self.output_dims = (
            output_dims if isinstance(output_dims, tuple) else (output_dims,)
        )
        self.config = config

        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.output_dims))
        else:
            self.register_buffer(
                '_zeros_template', torch.zeros(*self.output_dims)
            )
            self.bias = None

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        x = args[0]
        zeros = x.new_zeros(x.shape[0], *self.output_dims)
        if self.bias is not None:
            return zeros + self.bias
        return zeros
