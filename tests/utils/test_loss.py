from utils import loss
import pytest
from mock import Mock

class TestLoss:

    # @pytest.mark.xfail(raises=RuntimeError)    
    def test_compute_loss(self, mocker):
        p = 1
        targets = Mock()
        targets.device = 'test'
        model = Mock(
            module=Mock(
                model='test'
                )
            )
        #model.module.model[-1] = ''
        # model.model[-1] = ''
        mocker.patch('torch.zeros')
        tuple = loss.compute_loss(p, targets, model)