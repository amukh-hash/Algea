from typing import Optional, Dict
from datetime import datetime
import logging
from backend.app.core import config
from backend.app.options.types import OptionsMode, OptionsDecision, GateReasonCode
from backend.app.options.gate.context import OptionsContext
from backend.app.options.gate.gate import OptionsGate
from backend.app.options.strategy.strike_selector import StrikeSelector
from backend.app.options.execution.executor import OptionsExecutor
from backend.app.options.execution.noop_executor import NoopExecutor
from backend.app.options.execution.paper_executor import PaperExecutor
from backend.app.models.teacher_o_runner import TeacherORunner
from backend.app.options.data.iv_store import get_iv_store
from backend.app.options.data.chain_store import MockChainStore

logger = logging.getLogger(__name__)

class OptionsPod:
    def __init__(self):
        self.enabled = config.ENABLE_OPTIONS
        self.mode = OptionsMode(config.OPTIONS_MODE)
        
        if not self.enabled or self.mode == OptionsMode.OFF:
            self.executor = None
            return
            
        # Init Components
        self.iv_store = get_iv_store()
        self.chain_store = MockChainStore() # Default to mock for now
        self.teacher = TeacherORunner() # Mock
        self.gate = OptionsGate() # Uses default/tuned thresholds
        self.selector = StrikeSelector()
        
        # Init Executor
        if self.mode == OptionsMode.MONITOR:
            self.executor = NoopExecutor()
        elif self.mode == OptionsMode.PAPER:
            self.executor = PaperExecutor()
        else:
            self.executor = NoopExecutor() # Fallback
            
    def on_signal(self, ctx: OptionsContext) -> Optional[OptionsDecision]:
        if not self.enabled or self.mode == OptionsMode.OFF:
            return None
            
        # 1. Enrich Context (if missing data)
        if not ctx.iv_snapshot:
            ctx.iv_snapshot = self.iv_store.get_iv(ctx.ticker, ctx.timestamp, 30)
            
        # 2. Gate
        gate_decision = self.gate.evaluate(ctx)
        if not gate_decision.should_trade:
            # Log veto
            logger.info(f"Gate Veto: {gate_decision.reason_code} - {gate_decision.reason_desc}")
            return None
            
        # 3. Strategy / Strike Selection
        # We need 3D forecast from Student or Teacher? 
        # StrikeSelector takes Teacher Distribution usually.
        # But for now we might use Student or run Teacher.
        # Let's run Teacher mock.
        # (In real life, Teacher runs only if Gate passed Stage 2)
        
        # Mock input for Teacher
        dist = self.teacher.predict_distribution(None) # Mock ignores input
        
        # Fetch Chain
        chain = self.chain_store.get_chain(ctx.ticker, ctx.timestamp, "2023-02-17", ctx.underlying_price) # Mock expiry
        if not chain:
            logger.info("No chain found")
            return None
            
        candidate = self.selector.select_best_spread(ctx.underlying_price, chain, dist)
        
        if not candidate:
            logger.info("No valid spread found")
            return None
            
        # 4. Construct Decision
        decision = OptionsDecision(
            action="OPEN",
            candidate=candidate,
            quantity=1,
            reason="Gate passed + valid spread",
            timestamp=ctx.timestamp
        )
        
        return decision
        
    def execute(self, decision: OptionsDecision):
        if self.mode == OptionsMode.MONITOR:
            # Strict Guard
            if not isinstance(self.executor, NoopExecutor):
                 raise RuntimeError("Executor must be NoopExecutor in MONITOR mode")
            self.executor.execute(decision)
            
        elif self.mode == OptionsMode.PAPER:
            self.executor.execute(decision)
