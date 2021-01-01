from models import yolo
import pytest


class TestYolo:
    def test_detect(self):
        pass
        nc = 1
        anchors = {2}
        ch = 1
        #yolo.Detect(nc=nc, anchors=anchors, ch=ch)