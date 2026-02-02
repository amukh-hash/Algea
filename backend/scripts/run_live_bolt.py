"""
run_live_bolt.py - Production Inference Coordinator (Classification)

Orchestrates the entire inference pipeline:
1. Ingests live ticks
2. Generates features (including HMM regime)
3. Queries Student classifier
4. Validates with Judge (XGBoost)
5. Logs for Shadow Mode analysis

Usage:
    python run_live_bolt.py --shadow-mode  # Safe logging only
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import joblib
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.student import StudentModel

# --- CONFIGURATION ---
CONFIG = {
    'student_model_path': 'backend/models/bolt_student_v1.pt',
    'judge_model_path': 'backend/models/judge_xgboost_v1.pkl',
    'scaler_path': 'backend/models/scaler.pkl',
    'hmm_path': 'backend/models/hmm_regime.pkl',
    'feature_window': 60,      # Ticks required for feature calculation
    'judge_threshold': 0.60,   # Minimum confidence to execute
    'max_latency_ms': 100,     # Max allowable tick-to-trade time
}

# Setup Logging
log_dir = Path("backend/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "shadow_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BOLT_LIVE")


class BoltInferenceEngine:
    """
    Production inference engine with dual-model validation.
    Uses Classification paradigm: Student predicts [Neutral, Buy, Sell].
    """
    
    def __init__(self):
        logger.info("Initializing Bolt Inference Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Preprocessing Artifacts
        logger.info(f"Loading Scaler from {CONFIG['scaler_path']}...")
        try:
            self.scaler = joblib.load(CONFIG['scaler_path'])
        except FileNotFoundError:
            logger.warning("Scaler not found. Run prep_artifacts.py first!")
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
            self.scaler.fit(np.zeros((1, 5)))
        
        logger.info(f"Loading HMM from {CONFIG['hmm_path']}...")
        try:
            self.hmm = joblib.load(CONFIG['hmm_path'])
        except FileNotFoundError:
            logger.warning("HMM not found. Run prep_artifacts.py first!")
            self.hmm = None

        # 2. Load Student Model
        logger.info(f"Loading Student from {CONFIG['student_model_path']}...")
        try:
            checkpoint = torch.load(CONFIG['student_model_path'], map_location=self.device)
            self.student = StudentModel(checkpoint['config']).to(self.device)
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.student.eval()
            logger.info(f"Student loaded. Config: {checkpoint['config']}")
        except FileNotFoundError:
            logger.warning("Student model not found. Run train_nightly_distill.py first!")
            self.student = None

        # 3. Load Judge Model
        logger.info(f"Loading Judge from {CONFIG['judge_model_path']}...")
        try:
            self.judge = joblib.load(CONFIG['judge_model_path'])
        except FileNotFoundError:
            logger.warning("Judge not found. Run train_judge_advanced.py first!")
            self.judge = None
        
        # Buffer for feature calculation
        self.tick_buffer = []
        self.returns_buffer = []

    def calculate_features(self, tick_data):
        """Calculate features from tick buffer."""
        df = pd.DataFrame(self.tick_buffer)
        
        # Returns
        returns = df['price'].pct_change().fillna(0).values[-1]
        
        # Volatility (rolling 20)
        vol = df['price'].pct_change().rolling(20).std().fillna(0).values[-1]
        
        # RSI
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().fillna(0).values[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0.001).values[-1]
        rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # Spread
        spread = tick_data.get('ask', 100.05) - tick_data.get('bid', 99.95)
        
        # Regime
        if self.hmm is not None and len(self.returns_buffer) >= 20:
            recent_returns = np.array(self.returns_buffer[-20:]).reshape(-1, 1)
            regime = self.hmm.predict(recent_returns)[-1]
        else:
            regime = 0
        
        return np.array([returns, vol, rsi, spread, regime], dtype=np.float32)

    def preprocess(self, tick_data):
        """Converts raw tick -> Feature Sequence -> Scaled Tensor."""
        # Append new tick
        self.tick_buffer.append(tick_data)
        
        # Calculate return for HMM
        if len(self.tick_buffer) >= 2:
            ret = (tick_data['price'] - self.tick_buffer[-2]['price']) / self.tick_buffer[-2]['price']
            self.returns_buffer.append(ret)
        
        # Keep buffer fixed size
        if len(self.tick_buffer) > CONFIG['feature_window'] + 50:
            self.tick_buffer.pop(0)
            self.returns_buffer.pop(0)
            
        if len(self.tick_buffer) < CONFIG['feature_window']:
            return None  # Cold start

        # Build sequence of features
        feature_sequence = []
        for i in range(len(self.tick_buffer) - CONFIG['feature_window'], len(self.tick_buffer)):
            # Simplified: use current tick features for each timestep
            features = self.calculate_features(self.tick_buffer[i])
            feature_sequence.append(features)
        
        feature_array = np.array(feature_sequence)  # (seq_len, num_features)
        
        # Scale
        feature_array = self.scaler.transform(feature_array)
        
        # Convert to Tensor: [1, seq_len, features]
        tensor_input = torch.FloatTensor(feature_array).unsqueeze(0).to(self.device)
        
        return tensor_input

    def predict(self, tick_data):
        """Full inference pipeline: Student -> Judge -> Decision."""
        start_time = time.perf_counter()
        
        # 1. Preprocess
        tensor_input = self.preprocess(tick_data)
        
        if tensor_input is None:
            return None  # Warming up

        # 2. Student Inference (Classification)
        if self.student is not None:
            with torch.no_grad():
                logits = self.student(tensor_input)
                probabilities = torch.softmax(logits, dim=1)
                action = torch.argmax(probabilities).item()  # 0: Neutral, 1: Buy, 2: Sell
                confidence = probabilities[0][action].item()
        else:
            action = 0
            confidence = 0.5

        # 3. Judge Inference (The Gatekeeper)
        if self.judge is not None:
            # Judge features: last tick features + action + confidence
            last_features = tensor_input[0, -1, :].cpu().numpy()
            judge_input = np.hstack([last_features, [action, confidence]]).reshape(1, -1)
            judge_prob = self.judge.predict_proba(judge_input)[0][1]
        else:
            judge_prob = 0.7

        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        return {
            'action': action,  # 0=Neutral, 1=Buy, 2=Sell
            'confidence': confidence,
            'judge_score': judge_prob,
            'latency': latency,
            'regime': int(self.returns_buffer[-1] > 0.01) if self.returns_buffer else 0
        }


def run_shadow_mode(args):
    """Shadow mode: log signals without executing trades."""
    engine = BoltInferenceEngine()
    
    logger.info("Starting Shadow Mode Loop...")
    
    action_names = {0: 'NEUTRAL', 1: 'BUY', 2: 'SELL'}
    
    try:
        for i in range(200):  # Simulation loop
            # Mock Tick
            tick = {
                'price': 100.0 + np.cumsum([np.random.normal(0, 0.1)])[0] if i == 0 else 
                         engine.tick_buffer[-1]['price'] + np.random.normal(0, 0.1),
                'bid': 99.95 + np.random.normal(0, 0.01),
                'ask': 100.05 + np.random.normal(0, 0.01),
                'timestamp': datetime.now()
            }
            
            result = engine.predict(tick)
            
            if result:
                # Decision Logic
                decision = "HOLD"
                if result['action'] == 1 and result['judge_score'] > CONFIG['judge_threshold']:
                    decision = "🟢 BUY"
                elif result['action'] == 2 and result['judge_score'] > CONFIG['judge_threshold']:
                    decision = "🔴 SELL"
                
                # Drift Alert
                drift_alert = " [⚠️ LATENCY]" if result['latency'] > CONFIG['max_latency_ms'] else ""
                
                # Log
                log_entry = (
                    f"Tick {i:3d} | "
                    f"Action: {action_names[result['action']]:8s} | "
                    f"Conf: {result['confidence']:.2%} | "
                    f"Judge: {result['judge_score']:.2%} | "
                    f"Lat: {result['latency']:5.2f}ms"
                    f"{drift_alert}"
                )
                
                if decision != "HOLD":
                    logger.info(f"🚀 SIGNAL: {log_entry} → {decision}")
                else:
                    logger.debug(log_entry)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Shadow Mode stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALGAI v2.0 in Live/Shadow Mode")
    parser.add_argument('--shadow-mode', action='store_true', default=True,
                        help="Run without executing trades")
    args = parser.parse_args()

    if args.shadow_mode:
        logger.info("=" * 50)
        logger.info("RUNNING IN SHADOW MODE")
        logger.info("No trades will be executed. Logging signals only.")
        logger.info("=" * 50)
        run_shadow_mode(args)
    else:
        logger.warning("!!! LIVE TRADING NOT IMPLEMENTED !!!")
