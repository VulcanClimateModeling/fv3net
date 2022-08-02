import torch.optim as optim
import dataclasses
from typing import (
    Any,
    Mapping,
)


@dataclasses.dataclass
class OptimizerConfig:
    name: str = "AdamW"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def instance(self, params):
        cls = getattr(optim, self.name)

        kwargs = dict(params=params, **self.kwargs)

        return cls(**kwargs)


# @dataclasses.dataclass
# class LearningRateScheduleConfig:
#     name: str
#     kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

#     @property
#     def instance(self,params) -> torch.optim.lr_scheduler:
#         kwargs = dict(params=params,**self.kwargs)

#         cls = getattr(optim.lr_scheduler, self.name)
#         return cls(**self.kwargs)
