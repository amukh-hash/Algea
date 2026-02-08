from typing import List, Dict
import numpy as np

def calculate_sim_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {}
        
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls)
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    return {
        "total_pnl": total_pnl,
        "count": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": (win_rate * avg_win) + ((1-win_rate) * avg_loss)
    }
