import pytest
import os
from unittest.mock import MagicMock
from backend.app.engine.options_pod import OptionsPod
from backend.app.options.execution.noop_executor import NoopExecutor
from backend.app.options.execution.paper_executor import PaperExecutor

from backend.app.core import config
from backend.app.options.types import OptionsMode

def test_options_pod_monitor_mode(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_OPTIONS", True)
    # OptionsPod converts string to Enum in init, so we need to ensure config.OPTIONS_MODE matches
    monkeypatch.setattr(config, "OPTIONS_MODE", "monitor")
    
    pod = OptionsPod()
    assert isinstance(pod.executor, NoopExecutor)
    
def test_options_pod_paper_mode(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_OPTIONS", True)
    monkeypatch.setattr(config, "OPTIONS_MODE", "paper")
    
    pod = OptionsPod()
    assert isinstance(pod.executor, PaperExecutor)

def test_options_pod_off_mode(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_OPTIONS", False)
    
    pod = OptionsPod()
    assert pod.executor is None
