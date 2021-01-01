from models import common
import pytest
import torch
class TestCommon:
    """Test Common."""

    def test_autopad_returns_p_when_not_none(self):
        k = 1
        p = 2
        autopad = common.autopad(k, p)
        assert p == autopad

    def test_autopad_returns_padded_when_none(self):
        k = 1
        p = None
        expected = 0
        autopad = common.autopad(k, p)
        assert expected == autopad

class TestCommonDWConv:
    def test_common_dwconv_init(self):
        c1 = 1
        c2 = 1
        k = 1
        s = 1
        act = True
        
        # -Conv(\n
        # (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)\n
        # (bn): BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n
        # (act): Hardswish()\

        dwconv = common.DWConv(c1, c2, k, s, act)
        # assert class attribute matches
        assert (1, 1) == dwconv.conv.stride

    @pytest.mark.skip(reason="This raises out of bounds in torch.nn._calculate_fan_in_and_fan_out")
    @pytest.mark.xfail(raises=IndexError)
    def test_forward(self):
        c1 = torch.zeros(1, dtype=torch.long)
        c2 = 1
        k = 1
        s = 1
        act = True

        dwconv = common.DWConv(c1, c2, k, s, act)
        assert (1, 1) == dwconv.conv.stride

        x = 1
        hardswish = dwconv.forward(x)
        assert "test" == hardswish

    @pytest.mark.xfail(raises=IndexError)
    def test_fuseforward_raises_IndexError(self):
        c1 = torch.zeros(1, dtype=torch.long)
        c2 = 1
        k = 1
        s = 1
        act = True

        dwconv = common.DWConv(c1, c2, k, s, act)
        assert (1, 1) == dwconv.conv.stride

        x = 1
        hardswish = dwconv.fuseforward(x)
        assert "test" == hardswish
    
    @pytest.mark.xfail(raises=RuntimeError)
    def test_fuseforward_raises_ZeroDivisionError(self):
        c1 = torch.zeros(1, dtype=torch.long)
        c2 = torch.zeros(1, dtype=torch.long)
        k = 1
        s = 1
        act = True

        dwconv = common.DWConv(c1, c2, k, s, act)
        assert (1, 1) == dwconv.conv.stride

        x = torch.zeros(1, dtype=torch.long)
        hardswish = dwconv.fuseforward(x)
        assert "test" == hardswish


    @pytest.mark.xfail(raises=TypeError)
    def test_fuseforward_raises_TypeError(self):
        c1 = torch.ones(1, dtype=torch.float)
        c2 = torch.ones(1, dtype=torch.float)
        k = [1]
        s = 1
        act = True

        dwconv = common.DWConv(c1, c2, k, s, act)
        assert (1, 1) == dwconv.conv.stride

        x = torch.ones(1, dtype=torch.float)
        hardswish = dwconv.fuseforward(x)
        assert "test" == hardswish


    # common.Bottleneck
    # bn.forward
    # common.BottleneckCSP
    # bncsp.forward
    # common.C3
    # C3.forward
    # common.SPP
    # spp.forward
    # common.Focus
    # focus.forward
    # common.Concat
    # concat.forward
    # common.NMS
    # nms.forward
    # common.autoShape
    # autoshape.forward
    # common.Detections
    # detections.display
    # detections.print
    # detections.show
    # detections.save
    # detections.__len__
    # detections.tolist

    # common.Flatten
    # flatten.forward

    # common.Classify
    # classify.forward

