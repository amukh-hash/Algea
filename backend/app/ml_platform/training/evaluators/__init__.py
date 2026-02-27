from .tsfm_eval import pinball_loss


def evaluate_stub() -> dict:
    return {"sharpe": 1.1, "max_drawdown": 0.1, "calibration_score": 0.8}


__all__ = ["pinball_loss", "evaluate_stub"]
