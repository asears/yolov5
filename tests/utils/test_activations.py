from utils import activations
import pytest

class TestSiLU:
    def test_forward(self):
        pass

class TestHardswish:
    def test_forward(self):
        pass

class TestMemoryEfficientSwish:
    def test_forward(self):
        pass

class TestMemoryEfficientSwishF:
    def test_forward(self):
        pass

class TestMish:
    def test_forward(self):
        pass

class TestMemoryEfficientMish:
    def test_forward(self):
        pass

class TestMemoryEfficientMishF:
    def test_forward(self):
        pass

class TestFReLU:
    def test_forward(self):
        c1 = 1
        k = 3
        frelu = activations.FReLU(c1, k)
        x = 1
        frelu.forward(x)