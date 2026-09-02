################################################################################
# MaskedLinear block and PerforatedDendriticANN for the Poirazi variant.       #
################################################################################

#
"""
Imports
"""
import torch

from torch    import Tensor, nn
from torch.nn import functional as F
from typing   import List, Optional


#
"""
Model
"""
class MaskedLinear(nn.Module):
    '''
    Linear block whose connectivity is pinned by a boolean mask

    Notes:
        - The mask multiplies the weight inside forward instead of being
          baked into the weight values
            -> PAI builds every dendrite by deep copying its parent module
               and then overwriting all of its parameters with fresh noise,
               so a connectivity carried in the parameter values would not
               survive dendrite creation
            -> Buffers are left alone by that, so we implement that way
        - W <- W (*) M
            -> Evaluated every forward instead of after every gradient step
            -> The multiply is in the graph, so masked entries get a zero
               gradient already
        - _init=False skips reset_parameters so the factory can copy a parent
          without consuming any global torch RNG calls

    Signature:
        in_features (int):
            - Number of input features
        out_features (int):
            - Number of output features
        mask (Tensor):
            - Boolean connectivity, 1 wherever a synapse exists
                Shape -> [out_features, in_features]
    '''

    def __init__(
        self,
        in_features : int,
        out_features: int,
        mask        : Tensor,
        *,
        _init: bool = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('rf', mask.detach().clone().float())

        if _init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def synapse_count(self) -> int:
        return int(self.rf.sum().item())

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'synapses={self.synapse_count()}'
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.rf, self.bias)


class PerforatedDendriticANN(nn.Module):
    '''
    Dendritic ANN whose dendrites come from PAI, not from a layer

    Notes:
        - Each soma is a single MaskedLinear reading the input directly,
          and PAI wraps it so that the paper's dendrites become dendrite slots
            -> soma_j = f2(sum_k c_jk f1(W_jk x + b_jk) + b_j)
            -> W_jk   = dendrite_module.layers[k]
            -> f1     = LeakyReLU
            -> c_jk   = dendrites_to_top
            -> b_j    = MaskedLinear bias
            -> Original paper has no input to soma path
                - Soma masks is usually all zero
                - Only contributes bias

    Signature:
        input_size (int):
            - Number of flattened input features per sample
        num_layers (int):
            - Number of somatic layers
        soma (List[int]):
            - Somata for each layer
        num_classes (int):
            - Number of output classes
        name (str):
            - Model name used when building output paths
        soma_masks (List[Tensor]):
            - Input mask of each soma block, in layer order; only used when
              soma_modules is None
                num_layers masks, each Shape -> [soma[j], in_features]
        soma_modules (List[nn.Module]):
            - Pre-built soma modules, one per layer; when provided, soma_masks
              is ignored and these are registered directly
        relu_slope (float):
            - Negative slope of the leaky relu activations
        dropout (bool):
            - Whether a dropout op follows each activation
        rate (float):
            - Dropout probability, ignored when dropout is False
    '''
    def __init__(
        self,
        input_size  : int,
        num_layers  : int,
        soma        : List[int],
        num_classes : int,
        name        : str,
        soma_masks  : Optional[List[Tensor]] = None,
        soma_modules: Optional[List['nn.Module']] = None,
        relu_slope  : float = 0.1,
        dropout     : bool  = False,
        rate        : float = 0.0,
    ) -> None:
        super().__init__()
        self.name        = name
        self.num_classes = num_classes

        if soma_modules is None and (soma_masks is None or len(soma_masks) != num_layers):
            raise ValueError(
                f'Provide either soma_modules or soma_masks with {num_layers} entries.'
            )

        self.input  = nn.Identity()
        layer_names = ['input']

        in_features = input_size
        for j in range(num_layers):
            soma_name = f'soma_{j + 1}'

            if soma_modules is not None:
                setattr(self, soma_name, soma_modules[j])
            else:
                setattr(
                    self,
                    soma_name,
                    MaskedLinear(in_features, soma[j], soma_masks[j]),
                )
            layer_names.append(soma_name)

            setattr(
                self,
                f'{soma_name}_relu',
                nn.LeakyReLU(negative_slope = relu_slope),
            )
            layer_names.append(f'{soma_name}_relu')

            if dropout:
                setattr(self, f'{soma_name}_dropout', nn.Dropout(p = rate))
                layer_names.append(f'{soma_name}_dropout')

            in_features = soma[j]

        self.output = nn.Linear(in_features, num_classes)
        layer_names.append('output')

        self.layer_names = layer_names
        self.soma_names  = [f'soma_{j + 1}' for j in range(num_layers)]

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def layers(self) -> List[nn.Module]:
        return [getattr(self, n) for n in self.layer_names]

    def soma_modules(self) -> List[nn.Module]:
        return [getattr(self, n) for n in self.soma_names]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
