import pytest
from backend.app.options.strategy.vibe_check import vibe_check
from backend.app.options.data.types import IVSnapshot
from backend.app.options.types import GateReasonCode
from datetime import datetime

def test_vibe_check_pass():
    iv = IVSnapshot("AAPL", datetime.now(), 30, 0.2, iv_rank=0.5)
    passed, code = vibe_check(0.5, iv, "NORMAL", True)
    assert passed
    assert code == GateReasonCode.PASS

def test_vibe_check_liquidity():
    iv = IVSnapshot("AAPL", datetime.now(), 30, 0.2, iv_rank=0.5)
    passed, code = vibe_check(0.5, iv, "NORMAL", False)
    assert not passed
    assert code == GateReasonCode.REJECT_LIQUIDITY

def test_vibe_check_iv_rank():
    iv = IVSnapshot("AAPL", datetime.now(), 30, 0.2, iv_rank=0.01) # Low
    passed, code = vibe_check(0.5, iv, "NORMAL", True)
    assert not passed
    assert code == GateReasonCode.REJECT_IV
