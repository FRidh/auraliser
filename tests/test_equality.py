import pytest
import numpy as np
from auraliser.auralisation import equality

items = [
    'abc',
    100,
    np.arange(10),
    {'foo': 10,
     'bar': 20,
     },
    {'foo': np.arange(10),
     'bar': np.arange(10),
     },
    {'foo' : {'bar': np.arange(10)}},
    None,
    ]


@pytest.fixture(params=items)
def item(request):
    return request.param

def test_equality(item):


    assert equality(item, item)