


import torch
from torch import Tensor
from typing import Union




class StateActionNorm:
    """
    Normalization utility for State and Action spaces.
    Supports mapping to [0, 1] or [-1, 1] ranges with handling for zero-span dimensions.
    """
    def __init__(
        self,
        min_val: Union[float, Tensor],
        max_val: Union[float, Tensor],
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        target_range: str = '0_1',
    ):

        min_val = torch.as_tensor(min_val, dtype=dtype, device=device)
        max_val = torch.as_tensor(max_val, dtype=dtype, device=device)
        try:
            _ = torch.broadcast_shapes(min_val.shape, max_val.shape)
        except RuntimeError:
            raise ValueError("Shapes of `min_val` and `max_val` are not broadcastable.")

        range_ = max_val - min_val
 
        zero_span_mask = (min_val == 0) & (max_val == 0)

        if torch.any((max_val < min_val) & (~zero_span_mask)):
            raise ValueError("Found dimensions where max_val < min_val with non-zero span.")

        if target_range not in ['0_1', '-1_1']:
            raise ValueError("`target_range` must be either '0_1' or '-1_1'")

        self.min_val: Tensor = min_val
        self.max_val: Tensor = max_val
        self.range:   Tensor = range_
        self.eps: float = eps
        self.target_range: str = target_range
        self.zero_span_mask: Tensor = zero_span_mask  


        def normalize_0_1(x: Tensor) -> Tensor:

            x = x.to(dtype=min_val.dtype, device=min_val.device)
            y = (x - self.min_val) / (self.range + self.eps)

            if self.zero_span_mask.any():
                y = y.clone()
                y = torch.where(self.zero_span_mask, torch.zeros_like(y), y)
            return y

        def denormalize_0_1(xn: Tensor) -> Tensor:
            xn = xn.to(dtype=min_val.dtype, device=min_val.device)
            y = xn * self.range + self.min_val

            if self.zero_span_mask.any():
                y = y.clone()
                y = torch.where(self.zero_span_mask, torch.zeros_like(y), y)
            return y

        self._normalize_0_1 = normalize_0_1
        self._denormalize_0_1 = denormalize_0_1

        if self.target_range == '0_1':
            self.normalize = self._normalize_0_1
            self.denormalize = self._denormalize_0_1
        else:  # '-1_1'
            def normalize_m1_1(x: Tensor) -> Tensor:
                y01 = self._normalize_0_1(x)
                y = y01 * 2.0 - 1.0
                if self.zero_span_mask.any():
                    y = y.clone()
                    y = torch.where(self.zero_span_mask, -torch.ones_like(y), y)
                return y

            def denormalize_m1_1(xn: Tensor) -> Tensor:
                y01 = (xn + 1.0) / 2.0
                y = self._denormalize_0_1(y01)
                if self.zero_span_mask.any():
                    y = y.clone()
                    y = torch.where(self.zero_span_mask, torch.zeros_like(y), y)
                return y

            self.normalize = normalize_m1_1
            self.denormalize = denormalize_m1_1












