# ALGAI Trading Platform - Complete Codebase Walkthrough

## 1. SYSTEM OVERVIEW

**ALGAI** is an institutional-grade algorithmic trading platform featuring:
- **Teacher-Student Distillation**: Chronos T5 (Teacher) → Chronos Bolt (Student)
- **Advanced Signal Processing**: MODWT wavelets + Unscented Kalman Filter/Smoother
- **Live Trading**: Alpaca API integration with "Overlord" execution loop
- **Deep Learning**: PatchTST transformers with LoRA fine-tuning
- **Options Trading**: Vectorized execution with Greeks calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ALGAI PLATFORM V2.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐     ┌───────────────────┐     ┌──────────────────┐  │
│  │ TEACHER (Nightly) │     │ STUDENT (Live)    │     │ OVERLORD         │  │
│  │ Chronos T5 Large  │────►│ Chronos Bolt      │────►│ Alpaca Execution │  │
│  │ MODWT + UKS       │     │ Sliding Wavelet   │     │ Infinite Loop    │  │
│  │ Acausal Smooth    │     │ + UKF (Causal)    │     │                  │  │
│  └───────────────────┘     └───────────────────┘     └──────────────────┘  │
│           │                         │                         │             │
│           └─────────────────────────┴─────────────────────────┘             │
│                                     │                                       │
│                          ┌──────────▼──────────┐                           │
│                          │  DATABENTO CLIENT   │                           │
│                          │  Historical + Live  │                           │
│                          │  L2 (MBP-10) Data   │                           │
│                          └─────────────────────┘                           │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         FRONTEND (React)                              │ │
│  │  Dashboard │ StrategyBuilder │ ConfidenceGauges │ GreeksRadar │ PnL  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. PROJECT STRUCTURE

```
algai/
├── backend/
│   ├── app/                          # Core application (38 Python files)
│   │   ├── main.py                   # FastAPI entry point
│   │   ├── api/
│   │   │   ├── databento_client.py   # DataBento L2 data client
│   │   │   └── v1/endpoints/         # REST API (data, hybrid, xai)
│   │   ├── core/                     # Training infrastructure
│   │   ├── data/                     # Data handlers & Parquet I/O
│   │   ├── engine/                   # Inference pipeline & execution
│   │   ├── features/
│   │   │   ├── context.py            # Market context engineering
│   │   │   └── signal_processing.py  # Wavelet + Kalman filters
│   │   ├── math/                     # Black-Scholes Greeks
│   │   ├── models/
│   │   │   ├── patchtst.py           # HybridPatchTST model
│   │   │   ├── loss.py               # StudentTradingLoss
│   │   │   ├── train.py              # TrainRequest/Response
│   │   │   └── ...                   # layers, lora, fusion
│   │   ├── preprocessing/            # RevIN, Fractional Diff
│   │   ├── targets/                  # Triple Barrier labeling
│   │   ├── utils/                    # Chronos adapter
│   │   └── xai/                      # Attribution & visualization
│   ├── scripts/                      # 22 training/execution scripts
│   │   ├── train_nightly_distill.py  # Teacher-Student distillation
│   │   ├── run_live_bolt.py          # Live trading with Bolt
│   │   ├── run_overlord.py           # Alpaca execution loop
│   │   └── ...
│   └── tests/                        # 6 test files
├── frontend/                         # React application
│   └── src/components/               # Dashboard, StrategyBuilder, etc.
└── .env                              # Alpaca/DataBento credentials
```

---

## 3. CORE COMPONENTS

### 3.1 DataBento Client: `app/api/databento_client.py`

**Purpose**: High-frequency L2 market data ingestion

**Class**: `DatabentoClient`

| Method | Description |
|--------|-------------|
| `__init__(api_key, mock_mode)` | Initialize with API key or mock mode |
| `get_historical_range(symbol, start, end, schema)` | Fetch historical MBP-10 data |
| `start_live_stream(symbol, schema)` | Generator yielding live ticks |

**Mock Mode Features**:
- Generates random walk price paths
- Creates synthetic MBP-10 (10-level bid/ask) columns
- Simulates 50ms tick latency

**Production Mode**:
- Connects to GLBX.MDP3 (Globex futures) or XNAS.ITCH (Nasdaq)
- Returns pandas DataFrame with full order book depth

---

### 3.2 Signal Processing: `app/features/signal_processing.py`

**Purpose**: Advanced denoising for Teacher (acausal) and Student (causal)

#### Teacher: Acausal Smoothing (MODWT + UKS)

**Function**: `apply_modwt_uks(data, wavelet='db4', level=3)`

```
Input Signal
    │
    ▼
┌─────────────────┐
│ SWT Decompose   │  Stationary Wavelet Transform
│ (pywt.swt)      │  Multi-resolution analysis
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ cA     │ │ cD     │  Approximation & Detail coefficients
│ (keep) │ │ (soft  │
│        │ │ thresh)│
└────────┘ └────────┘
         │
         ▼
┌─────────────────┐
│ ISWT Reconstruct│  Inverse SWT → Denoised signal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ UKF Forward     │  Unscented Kalman Filter (batch)
│ Pass            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RTS Backward    │  Rauch-Tung-Striebel Smoother
│ Smoother        │  (acausal - uses future data)
└────────┬────────┘
         │
         ▼
    Smoothed Signal
```

**State Model**:
```
x = [price, velocity]
fx(x, dt) = [price + velocity*dt, velocity]
hx(x) = [price]
```

#### Student: Causal Filtering (Sliding Wavelet + UKF)

**Function**: `apply_sliding_wavelet_ukf(data_window, ukf_object)`

- Uses `pywt.wavedec` (standard DWT, not stationary)
- Applies soft thresholding to detail coefficients
- Single UKF predict/update step (no future lookahead)
- Returns filtered price and updated UKF state

#### Trend Scanning (Numba-accelerated)

**Function**: `trend_scanning_labels(prices, window_min, window_max)`

- `@jit(nopython=True)` for speed
- Computes t-statistic of linear regression slope
- Returns label based on max |t-stat| across windows

#### Triple Barrier Method

**Function**: `triple_barrier_labels(prices, volatility, vertical_barrier_window, sl_tp_ratio)`

- Upper barrier: price × (1 + k × σ)
- Lower barrier: price × (1 - k × σ / sl_tp_ratio)
- Returns labels: 1 (Buy), -1 (Sell), 0 (Neutral)

---

### 3.3 Student Trading Loss: `app/models/loss.py`

**Purpose**: Multi-task loss for Teacher-Student distillation

**Class**: `StudentTradingLoss(nn.Module)`

**Three Loss Components**:

| Task | Loss | Purpose |
|------|------|---------|
| 1. Distillation | KL Divergence | Match Teacher's output distribution |
| 2. Sortino Ratio | -E[Sortino] | Maximize risk-adjusted returns |
| 3. Focal | Focal BCE | Focus on hard trading opportunities |

**Homoscedastic Weighting**:
```python
# Learnable task weights via uncertainty
self.log_vars = nn.Parameter(torch.zeros(3))

# Combined loss
precision = exp(-log_var)
total = precision * loss + log_var  # for each task
```

**Sortino Calculation**:
```
1. probs = softmax(logits)
2. expected_prices = sum(probs × bin_centers)
3. returns = diff(expected_prices) / expected_prices[:-1]
4. downside_std = sqrt(mean(clamp(returns - rfr, max=0)²))
5. sortino = mean(returns) / downside_std
6. loss = -mean(sortino)  # Maximize by minimizing negative
```

---

### 3.4 Training Request Model: `app/models/train.py`

**Pydantic models for training API**:

```python
class TrainRequest(BaseModel):
    # Data
    data_source: str = "synthetic"
    symbol: str = "AAPL"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    timeframe: str = "1d"

    # Model
    model_type: str = "patchtst"
    lookback_window: int = 64
    forecast_horizon: int = 16
    patch_len: int = 8
    stride: int = 4
    d_model: int = 64

    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3

class TrainResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]
    training_time: float
```

---

### 3.5 Nightly Distillation Script: `scripts/train_nightly_distill.py`

**Purpose**: Transfer knowledge from Chronos T5 Large → Bolt Small

**Workflow**:

```
1. DATA PREP
   DatabentoClient (mock) → prices → volatility → TBM labels

2. TEACHER INFERENCE
   Load Chronos T5 Large
   For each window:
     - Tokenize context (64 steps)
     - Forward pass with labels
     - Store logits
   Delete teacher (free GPU memory)

3. STUDENT TRAINING
   Load Chronos Bolt Small
   Create StudentTradingLoss with bin centers

   For each batch:
     - Student forward (input_ids → logits)
     - Calculate loss(student_logits, teacher_logits, tbm_labels)
     - Backprop & update

   Save distilled model
```

**DistillationDataset**:
- Returns: input_ids, teacher_logits, tbm_label
- Context: 64 tokens
- Prediction: 32 tokens

---

### 3.6 Live Bolt Execution: `scripts/run_live_bolt.py`

**Purpose**: Real-time trading with distilled Student model

**Components**:

| Function | Description |
|----------|-------------|
| `perform_daytime_update()` | SimTS-style contrastive adaptation |
| `get_meta_confidence()` | Placeholder for Judge model |
| `allocate_portfolio()` | Half-Kelly position sizing |
| `run_live()` | Main execution loop |

**Execution Flow**:
```
1. Initialize Bolt model + DatabentoClient
2. For each tick:
   - Append to sliding window (deque, maxlen=64)
   - Apply causal filter (sliding wavelet + UKF)
   - Chronos inference → predicted price
   - Meta confidence → Kelly allocation
   - Generate signal: BUY/SELL/WAIT
   - Every 10 ticks: SimTS adaptation (cosine loss on encoder)
```

**SimTS Adaptation**:
```python
# Create augmented views
v1 = window * (1 + noise1)
v2 = window * (1 + noise2)

# Encode both
emb1 = encoder(tokenize(v1)).mean(dim=1)
emb2 = encoder(tokenize(v2)).mean(dim=1)

# Maximize similarity
loss = 1 - cosine_similarity(emb1, emb2)
loss.backward()
```

---

### 3.7 Overlord Execution: `scripts/run_overlord.py`

**Purpose**: Production trading loop with Alpaca API

**Class**: `Overlord`

**Initialization**:
- Loads Alpaca credentials from `.env`
- Creates TradingClient + StockHistoricalDataClient
- Verifies account (buying power, trading status)
- Loads Physics Engine (Chronos) and Judge (XGBoost)

**Tick Loop** (every 60s):
```
1. Check if market open (Alpaca Clock)
2. For each ticker in [NVDA, SPY, QQQ, IWM, TSLA]:
   a. Fetch last 600 1-minute bars
   b. Physics inference → direction + confidence
   c. Judge review (if available)
   d. Position check
   e. Execute order if signal confident
```

**Order Execution**:
```python
req = MarketOrderRequest(
    symbol=ticker,
    qty=1,  # Fixed for safety
    side=OrderSide.BUY/SELL,
    time_in_force=TimeInForce.DAY
)
trading_client.submit_order(req)
```

---

## 4. API LAYER

### 4.1 FastAPI Entry Point: `app/main.py`

```python
app = FastAPI(title="Algo Trading Backend")

# CORS for React frontend
origins = ["http://localhost:3000", "http://localhost:3001"]

# Routes
app.include_router(data.router, prefix="/api/v1/data")
app.include_router(xai.router, prefix="/api/v1/xai")
app.include_router(hybrid.router, prefix="/api/v1/hybrid")
```

### 4.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/data/ingest` | POST | Fetch OHLCV (yfinance/synthetic) |
| `/api/v1/data/preprocess` | POST | Apply fractional differentiation |
| `/api/v1/hybrid/run` | POST | Full inference + trading pipeline |
| `/api/v1/xai/explain` | POST | Feature attribution + attention |

---

## 5. DATA LAYER

### 5.1 Abstract Base: `app/data/base.py`

```python
class DataHandler(ABC):
    @abstractmethod
    def fetch_data(symbol, start_date, end_date, timeframe) -> pd.DataFrame

    def validate_schema(df) -> bool:
        # Requires: open, high, low, close, volume
        # Index: DatetimeIndex
```

### 5.2 Data Handlers

| Module | Purpose |
|--------|---------|
| `data/yfinance_loader.py` | Yahoo Finance fetcher |
| `data/synthetic.py` | GBM price generator |
| `data/parquet_manager.py` | Partitioned Parquet I/O |
| `data/schema.py` | ThetaData-compatible options schema |

### 5.3 Parquet Manager

**ParquetOptionsLoader**:
- Path: `ticker={SYMBOL}/year={YYYY}/month={MM}.parquet`
- Chunked loading with date filtering

**SyntheticOptionsGenerator**:
- Generates ~1000 contracts/day
- Fields: strike, expiration, bid, ask, Greeks, open interest

---

## 6. PREPROCESSING LAYER

### 6.1 Fractional Differentiation: `app/preprocessing/fractional.py`

**Purpose**: Make price series stationary while preserving memory

**FFD Weight Formula**:
```
w_k = -w_{k-1} / k × (d - k + 1)
```

**Parameters**:
- `d`: Differentiation order (0-1, typically 0.4)
- `thres`: Weight cutoff threshold (default 1e-4)

### 6.2 RevIN: `app/preprocessing/revin.py`

**Purpose**: Reversible Instance Normalization

```python
# Normalize
x_norm = (x - mean) / std
x_norm = gamma * x_norm + beta  # learnable affine

# Denormalize
x = (x_norm - beta) / gamma * std + mean
```

---

## 7. MODELS LAYER

### 7.1 HybridPatchTST: `app/models/patchtst.py`

**Architecture**:
```
Input (B, L, F)
    │
    ▼
┌─────────┐
│ RevIN   │  Normalize per feature
└────┬────┘
     ▼
┌─────────────────┐
│ PatchEmbedding  │  Unfold into patches, project to d_model
└────────┬────────┘
         ▼
┌─────────────────┐
│ PatchTSTBackbone│  Transformer encoder (2 layers, 4 heads)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Global Flatten  │  Reshape to (B, F×N×d_model)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Direction│ │Volatility│
│Head     │ │Head      │
│(3 class)│ │(regression)│
└────────┘ └──────────┘
```

**Parameters**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_input_features` | - | OHLCV features |
| `lookback_window` | - | Historical sequence |
| `patch_len` | 16 | Patch size |
| `stride` | 8 | Overlap stride |
| `d_model` | 128 | Embedding dim |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Transformer layers |
| `use_lora` | False | LoRA fine-tuning |

### 7.2 Custom Layers: `app/models/layers.py`

**Time2Vec**:
```
t2v(τ)[0] = w_0 × τ + φ_0           (linear)
t2v(τ)[i] = sin(w_i × τ + φ_i)      (periodic, i > 0)
```

**PatchEmbedding**:
```
Input (B, T, C) → Permute → Unfold → (B, C, N, P) → Linear → (B, C, N, d_model)
```

### 7.3 LoRA: `app/models/lora.py`

**Formula**:
```
output = W×x + (B @ A) × (α/r) × x
```

- A: (rank, in_features) - Kaiming init
- B: (out_features, rank) - Zero init
- Original layer frozen

### 7.4 Fusion: `app/models/fusion.py`

**GatedMultimodalUnit**:
```
z = sigmoid(W_z × [E_text, E_price])
H = z × tanh(W_t × E_text) + (1-z) × tanh(W_p × E_price)
```

---

## 8. ENGINE LAYER

### 8.1 InferencePipe: `app/engine/pipe.py`

**Purpose**: Async GPU inference + CPU execution

```
ThreadPool IO (2 workers)
    │
    ├── Prefetch next day's data
    │
    ▼
GPU Inference (HybridPatchTST)
    │
    ├── direction_prob, volatility
    │
    ▼
CPU Execution (VectorizedExecutioner)
    │
    └── select_contracts, update_positions, check_exits
```

### 8.2 VectorizedExecutioner: `app/engine/vectorized.py`

**Position Management**:
```python
columns = [
    "entry_date", "root", "expiration", "strike", "right",
    "entry_price", "quantity", "current_price", "pnl",
    "status", "max_pnl_pct"
]
```

**Exit Conditions**:
| Condition | Threshold | Label |
|-----------|-----------|-------|
| Take Profit | pnl_pct ≥ 50% | "TP" |
| Stop Loss | pnl_pct ≤ -20% | "SL" |
| Expiration | date ≥ expiration | "EXP" |

---

## 9. CORE TRAINING

### 9.1 Loss Functions: `app/core/loss.py`

**FocalLoss**:
```
FL(p_t) = -α × (1 - p_t)^γ × log(p_t)
```

**UniversalLoss**:
```
total = FocalLoss(direction) + weight × MSE(volatility)
```

### 9.2 Trainer: `app/core/trainer.py`

- Mixed precision (fp16) via `torch.cuda.amp`
- Gradient clipping (max_norm=4.0)
- Early stopping with patience
- Progress bar (tqdm)

---

## 10. XAI LAYER

### 10.1 Attribution: `app/xai/attribution.py`

**FeatureAttributor**:
- Uses Captum's Integrated Gradients
- Baseline: zeros
- Returns attribution tensor

### 10.2 Visualization: `app/xai/visualization.py`

- Mock attention weights (placeholder)
- Shape: (B, heads, num_patches, num_patches)

---

## 11. TARGETS LAYER

### 11.1 Triple Barrier: `app/targets/triple_barrier.py`

**Functions**:

| Function | Description |
|----------|-------------|
| `get_daily_vol(close, span)` | EWM volatility |
| `apply_triple_barrier(...)` | Barrier labeling |
| `get_purged_indices(labels_df)` | Non-overlapping trades |

**Labels**: 0 (Neutral), 1 (Buy), 2 (Sell)

---

## 12. MATH LAYER

### 12.1 Greeks: `app/math/greeks.py`

**Function**: `black_scholes_vectorized(S, K, T, r, sigma, option_type)`

**Formulas**:
```
d1 = [ln(S/K) + (r + σ²/2)×T] / (σ×√T)
d2 = d1 - σ×√T

Call = S×N(d1) - K×e^(-rT)×N(d2)
Delta_call = N(d1)
Gamma = n(d1) / (S×σ×√T)
Vega = S×n(d1)×√T / 100
```

---

## 13. DATA FLOW DIAGRAMS

### 13.1 Nightly Distillation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NIGHTLY DISTILLATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐             │
│  │ DataBento   │───────►│ MODWT + UKS │───────►│ Triple      │             │
│  │ Historical  │ L2     │ (Acausal)   │ Clean  │ Barrier     │             │
│  │ MBP-10      │ Data   │             │ Signal │ Labels      │             │
│  └─────────────┘        └─────────────┘        └──────┬──────┘             │
│                                                        │                    │
│  ┌─────────────┐                                      │                    │
│  │ Chronos T5  │                                      │                    │
│  │ Large       │──────────────────────────────────────┼───────┐            │
│  │ (Teacher)   │                                      │       │            │
│  └──────┬──────┘                                      │       │            │
│         │                                             │       │            │
│         │ Logits                                      │       │            │
│         │                                             │       │            │
│         ▼                                             ▼       │            │
│  ┌──────────────────────────────────────────────────────────┐ │            │
│  │                    STUDENT TRAINING                       │ │            │
│  │  ┌────────────────────────────────────────────────────┐  │ │            │
│  │  │              StudentTradingLoss                     │  │ │            │
│  │  │                                                     │  │ │            │
│  │  │  KL(Student || Teacher)  +  -Sortino  +  Focal     │  │◄┘            │
│  │  │         ↑                      ↑           ↑       │  │              │
│  │  │    student_logits        expected_prices  tbm_labels│  │              │
│  │  └────────────────────────────────────────────────────┘  │              │
│  │                           │                               │              │
│  │                           ▼                               │              │
│  │                   Chronos Bolt Small                     │              │
│  │                      (Student)                           │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Live Trading Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVE TRADING LOOP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ DataBento   │                                                            │
│  │ Live Stream │───┐                                                        │
│  │ (Ticks)     │   │                                                        │
│  └─────────────┘   │                                                        │
│                    ▼                                                        │
│            ┌──────────────┐                                                 │
│            │ Sliding      │                                                 │
│            │ Window       │                                                 │
│            │ (64 ticks)   │                                                 │
│            └──────┬───────┘                                                 │
│                   │                                                         │
│         ┌─────────┴─────────┐                                              │
│         ▼                   ▼                                              │
│  ┌─────────────┐     ┌─────────────┐                                       │
│  │ Sliding     │     │ Chronos     │                                       │
│  │ Wavelet     │     │ Bolt        │                                       │
│  │ + UKF       │     │ Inference   │                                       │
│  │ (Causal)    │     │             │                                       │
│  └──────┬──────┘     └──────┬──────┘                                       │
│         │                   │                                               │
│         │ filtered_price    │ predicted_price                              │
│         │                   │                                               │
│         └─────────┬─────────┘                                              │
│                   ▼                                                         │
│            ┌──────────────┐                                                 │
│            │ Judge        │                                                 │
│            │ (XGBoost)    │                                                 │
│            └──────┬───────┘                                                 │
│                   │ confidence                                              │
│                   ▼                                                         │
│            ┌──────────────┐                                                 │
│            │ Kelly        │                                                 │
│            │ Allocation   │                                                 │
│            └──────┬───────┘                                                 │
│                   │ position_size                                          │
│                   ▼                                                         │
│            ┌──────────────┐                                                 │
│            │ BUY / SELL   │────────► Alpaca API                            │
│            │ / WAIT       │                                                 │
│            └──────────────┘                                                 │
│                                                                             │
│  [Every 10 ticks]                                                          │
│            ┌──────────────┐                                                 │
│            │ SimTS        │                                                 │
│            │ Adaptation   │──────────► Update Encoder                      │
│            │ (Cosine Loss)│                                                 │
│            └──────────────┘                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. KEY ALGORITHMS

### 14.1 MODWT + UKS (Teacher Signal)

```
Input: Raw price series (can use future data)

1. Stationary Wavelet Transform (SWT)
   - Level 3 decomposition with db4 wavelet
   - Preserves original signal length

2. Soft Thresholding
   - σ = median(|cD|) / 0.6745
   - threshold = σ × sqrt(2 × log(n))
   - cD_new = sign(cD) × max(0, |cD| - threshold)

3. Inverse SWT
   - Reconstruct denoised signal

4. Unscented Kalman Filter (Forward)
   - State: [price, velocity]
   - Uses sigma points for nonlinear dynamics

5. RTS Smoother (Backward)
   - Incorporates future information
   - Produces optimal estimate given ALL data
```

### 14.2 Sliding Wavelet + UKF (Student Signal)

```
Input: Price window (causal - no future data)

1. Standard DWT (wavedec)
   - Level 2 decomposition
   - Symmetric extension mode

2. Soft Thresholding
   - Same as teacher but on DWT coefficients

3. Inverse DWT
   - Get reconstructed value at window end

4. UKF Single Step
   - predict() using state transition
   - update() with new observation
   - Returns current filtered estimate
```

### 14.3 Half-Kelly Allocation

```python
def allocate_portfolio(confidence):
    p = confidence  # Probability of success
    if p < 0.5:
        return 0.0  # Don't bet on losing odds

    b = 2.0  # Payoff ratio (win/loss)
    q = 1.0 - p

    kelly = (p * b - q) / b
    return max(0.0, kelly * 0.5)  # Half-Kelly for safety
```

---

## 15. FRONTEND

### 15.1 Components

| Component | Purpose |
|-----------|---------|
| `App.js` | Root component |
| `Dashboard.js` | Main trading interface |
| `StrategyBuilder.js` | Visual strategy editor (ReactFlow) |
| `ConfidenceGauges.js` | Direction/volatility pie charts |
| `GreeksRadar.js` | Delta/gamma/theta/vega radar |
| `PnLCurve.js` | Profit/loss line chart |

### 15.2 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                 Institutional Hybrid Engine                  │
│                      [Run Analysis]                          │
├───────────────────┬───────────────────┬─────────────────────┤
│ The Why (AI)      │ The What (Trade)  │ The How (Lifecycle) │
│                   │                   │                     │
│ ConfidenceGauges  │ Contract Details  │ PnLCurve            │
│ - Direction %     │ - Root            │                     │
│ - Volatility σ    │ - Strike/Right    │ Trade Log           │
│                   │ - Entry Price     │ - Date              │
│ Attention Heatmap │                   │ - Symbol            │
│ (placeholder)     │ GreeksRadar       │ - PnL               │
│                   │                   │                     │
└───────────────────┴───────────────────┴─────────────────────┘
```

---

## 16. CONFIGURATION

### 16.1 Environment Variables (`.env`)

```
ALPACA_API_KEY=<your_key>
ALPACA_SECRET_KEY=<your_secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DATABENTO_API_KEY=<optional>
```

### 16.2 Dependencies

**Python (requirements.txt)**:
- fastapi, uvicorn
- torch, transformers
- chronos-forecasting
- pywt (wavelets)
- filterpy (Kalman)
- numba (JIT)
- alpaca-py
- xgboost
- riskfolio-lib

**Frontend (package.json)**:
- react 19, react-dom
- reactflow, recharts
- axios

---

## 17. EXECUTION COMMANDS

```bash
# Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm start

# Nightly Distillation
python backend/scripts/train_nightly_distill.py

# Live Trading (Bolt)
python backend/scripts/run_live_bolt.py

# Overlord (Alpaca)
python backend/scripts/run_overlord.py
```

---

## 18. FILE STATISTICS

| Category | Count |
|----------|-------|
| Backend app modules | 38 |
| Training scripts | 22 |
| Test files | 6 |
| Frontend components | 5 |
| Total source files | 88 |

---

## 19. SCRIPTS REFERENCE

### Training Scripts

| Script | Purpose |
|--------|---------|
| `train_nightly_distill.py` | Teacher-Student distillation |
| `train_chronos_lora.py` | Chronos LoRA fine-tuning |
| `train_chronos_phase2.py` | Chronos phase 2 training |
| `train_universal.py` | Universal multi-task model |
| `train_global.py` | Global cross-stock model |
| `train_ensemble.py` | Ensemble stacking |
| `train_specialist.py` | Per-stock specialists |
| `train_judge_advanced.py` | XGBoost meta-labeler |
| `train_metalabeler.py` | Meta-labeling model |
| `train_stacking.py` | Stacking ensemble |
| `train_finetune.py` | General fine-tuning |
| `train_teacher_t5.py` | T5 teacher training |

### Execution Scripts

| Script | Purpose |
|--------|---------|
| `run_live_bolt.py` | Live trading with Bolt |
| `run_overlord.py` | Alpaca execution loop |
| `predict_ensemble.py` | Ensemble inference |
| `backtest_ensemble.py` | Backtesting |

### Data Scripts

| Script | Purpose |
|--------|---------|
| `download_alpaca.py` | Fetch Alpaca data |
| `phase1_feature_engineering.py` | Feature extraction |
| `generate_judge_data.py` | Judge training data |
| `unzip_data.py` | Data decompression |
| `continuous_learning.py` | Online retraining |
| `orchestrate_overnight.py` | Overnight batch jobs |

---

## 20. TEST SUITE

| Test File | Coverage |
|-----------|----------|
| `test_data.py` | Data ingestion |
| `test_greeks.py` | Black-Scholes |
| `test_hybrid.py` | Hybrid pipeline |
| `test_preprocessing.py` | FFD, RevIN |
| `test_signal_processing.py` | Wavelet + Kalman |
| `test_vectorized_lifecycle.py` | Execution engine |

---

*Last Updated: January 2026*
