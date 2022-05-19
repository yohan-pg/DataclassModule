# datamodule

dataclass + nn.Module

## Usage 

```python
from datamodule import DataModule

class ExampleModule(DataModule):
    x: int
    
    def __post_init__(self):
        self.layer = nn.Linear(2, 2)
```