from datamodule import DataModule

import torch
import torch.nn as nn


def test_dataclass_fields_are_added():
    class ExampleModule(DataModule):
        x: int

    try:
        ExampleModule() # type: ignore
    except TypeError:
        pass
    else:
        assert False

    assert isinstance(ExampleModule(x=1), ExampleModule)


def test_nn_module_is_initialized():
    class ExampleModule(DataModule):
        def __post_init__(self):
            self.layer = nn.Linear(2, 2)

        def forward(self, x):
            return self.layer(x)

    ExampleModule()(torch.randn(1, 2))


def test_parameters_are_registered_properly():
    class ExampleModule(DataModule):
        def __post_init__(self):
            self.param = nn.parameter.Parameter(torch.tensor([0.0]))

    assert len(list(ExampleModule().parameters())) == 1


def test_repr_method_is_from_dataclass_and_str_method_is_from_nn_module():
    class ExampleModule(DataModule):
        __qualname__ = "ExampleModule" # avoids local scope name alteration 
        y: bool = False

        def __post_init__(self):
            self.layer = nn.Linear(2, 2)

    assert repr(ExampleModule()) == "ExampleModule(y=False)", repr(ExampleModule())
    assert str(ExampleModule()) == nn.Module.__repr__(ExampleModule())