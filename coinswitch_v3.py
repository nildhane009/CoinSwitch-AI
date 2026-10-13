"""
coinswitch_v5.py  -  CoinSwitch PRO futures bot (EXCHANGE_2)

What is new vs v4 (see the top-of-file notes next to each setting):
  1. API rate-limit guard   : every REST call goes through a per-endpoint
     sliding-window limiter built from CoinSwitch's published futures limits
     (per API key), with 429 back-off + jitter and a live "API Load" line.
  2. Watchlist scaling      : MAX_WATCHLIST_SYMBOLS cap, background candle
     seeding (KLines is only 30 req/60s), no eager leverage calls
     (Update Leverage is only 10 req/60s), stale-symbol pruning.
  3. WebSocket safety       : streams are re-subscribed after a reconnect and
     a setup is ignored if the 3 candles are not contiguous (gap protection).
  4. Order safety           : client_order_id on every market order and HTTP
     5xx is treated as "unknown outcome" (never as "failed").
  5. AI v3                  : signal-aligned shadow samples, funding / OI /
     24h-change / relative-strength / symbol-form features, walk-forward
     validation and an out-of-time tuned veto cut-off. AI can only SKIP.
  6. Risk gate              : daily-loss stop, loss-streak pause and a cap on
     same-direction positions (entries only; exits are never blocked).

Trading is OFF until you enable it with the keyboard controls, exactly as in v4.
"""
import builtins
import os
import re
import json
import math
import time
import shutil
import textwrap
import threading
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import socketio
from cryptography.hazmat.primitives.asymmetric import ed25519


# ============================================================
# CONFIGURATION
# ============================================================

INTERVAL = "5"          # 5-minute candles

# ================================================================
# MASTER SAFETY SWITCH
#
#   False -> paper trading only. Signals, SL, exits, PnL are all
#            simulated in memory. NOTHING is sent to CoinSwitch.
#   True  -> REAL orders are placed with REAL money on your
#            CoinSwitch PRO futures account (fixed 10x leverage,
#            MARKET orders). Do not flip this to True until you
#            have watched the bot run in paper mode and are
#            comfortable with its behaviour.
# ================================================================
# ============================================================
# ACCOUNT / EXECUTION MODES
# ============================================================
# PAPER_MODE:
#   True  -> simulated trades using paper_capital
#   False -> paper trading disabled
#
# LIVE_MODE:
#   True  -> real CoinSwitch Futures orders
#   False -> live execution disabled
#
# ORDERS_ENABLED:
#   True  -> execution is allowed according to the selected mode
#   False -> no order execution at all
#
# PAPER_MODE and LIVE_MODE must never be True together.
# ============================================================

ORDERS_ENABLED = False
PAPER_MODE = False
LIVE_MODE = False
BOT_SHUTTING_DOWN = False

# ============================================================
# AI MODE (local learning model v2 - no API key, runs inside this file)
#
#   AI_MODE False -> model only LEARNS from closed trades (passive).
#   AI_MODE True  -> once enough trades are learned, entries whose
#                    predicted win probability is clearly below the
#                    bot's own average win rate are VETOED. Before
#                    that point it is advisory only (never blocks).
#
# The AI can only SKIP an entry. It never opens a trade, never
# touches stop-losses, and never overrides any risk rule.
# Toggle at runtime with the A key.
# ============================================================
AI_MODE = True
AI_MIN_TRADES_TO_VETO = 150   # setup samples (real trades + signal-aligned shadow) needed before it may veto
AI_MIN_CLASS_COUNT = 20       # at least this many setup wins AND losses
AI_VETO_RATIO = 0.85          # veto if win-prob < avg win rate * this
AI_REQUIRE_VALID_EDGE = True  # veto only if validation shows real edge
AI_MAX_SAMPLES = 5000         # samples kept (random shadow dropped first)
AI_TRAIN_MAX = 2000           # most recent samples used per training run
AI_MIN_TRAIN_SAMPLES = 20
AI_TRAIN_EPOCHS = 80
AI_LEARNING_RATE = 0.4
AI_L2 = 0.02
AI_GBM_ROUNDS = 60
AI_GBM_LR = 0.15
AI_GBM_BINS = 10
AI_VAL_FRACTION = 0.2         # newest 20% = out-of-time validation
AI_MIN_VALID_TOTAL = 90
AI_MIN_VALID_AUC = 0.57
AI_RECENCY_HALFLIFE = 1200    # samples; older samples count less
AI_SHADOW_WEIGHT = 0.15       # RANDOM-time shadow weight (setup/real trade = 1.0)
AI_RETRAIN_MIN_SECONDS = 240  # throttle for shadow-driven retraining
AI_RETRAIN_MIN_NEW = 15       # new samples needed to retrain
AI_FLUSH_SECONDS = 30
# Shadow learner: PASSIVE simulator (no orders ever). Every
# AI_SHADOW_MIN_GAP_SEC per symbol+side it records features and
# later simulates the bot's SL + trailing + fees for the outcome.
AI_SHADOW_ENABLED = True
AI_SHADOW_INTERVAL = 2.0
AI_SHADOW_MIN_GAP_SEC = 300
AI_SHADOW_HORIZON_SEC = 3600
AI_SHADOW_MAX_PENDING = 400
AI_SHADOW_MAX_NEW_PER_CYCLE = 6

# --- AI v3 -------------------------------------------------------
# Signal-aligned shadow: every REAL crossover setup on a watched symbol
# (even ones the bot skipped: scanner gate, capital, slot, funding...) is
# simulated with the bot's own SL + trailing + fees. These samples come
# from the SAME distribution the model will be asked to judge, so the
# model no longer has to wait for 100+ real trades.
AI_SETUP_SHADOW_ENABLED = True
AI_SETUP_SHADOW_WEIGHT = 0.8     # weight of a simulated setup (real trade = 1.0)
AI_SETUP_MAX_PENDING = 300
AI_RANDOM_SHADOW_MAX_TRAIN = 600 # cap of random-time shadow rows per training run
# Walk-forward validation: expanding window over the newest 45% of rows.
AI_WF_FOLDS = 3
AI_WF_TEST_FRACTION = 0.45
AI_WF_MIN_TRAIN = 40
AI_WF_FOLD_MAX_ROWS = 1200
# Veto cut-off is TUNED out-of-time (maximise net ROI of kept trades) and
# must also show a lift on a held-out half before AI is allowed to veto.
AI_MIN_TUNE_ROWS = 40
AI_TUNE_MIN_KEEP = 0.35          # never veto more than 65% of setups
AI_TUNE_MIN_LIFT_ROI = 0.25      # mean ROI% per trade gained on the hold-out half

if PAPER_MODE and LIVE_MODE:
    raise RuntimeError("PAPER_MODE and LIVE_MODE cannot both be True.")



# Leverage applied to every symbol before its first trade (fixed,
# per your instruction). Must be <= the symbol's max_leverage -
# since watchlist_refresh_loop() only keeps symbols whose
# max_leverage is >= TRADE_LEVERAGE, this fixed value is always valid.
TRADE_LEVERAGE = 10

# Maximum allowed loss on trading capital when the fixed SL is hit.
# Raw price risk is multiplied by TRADE_LEVERAGE. >30% is rejected.
MAX_SL_RISK_PCT = 30.0

# LIVE protection / recovery safety
EMERGENCY_SL_PCT = 10.0

# ROI-based protection rules.
PRIMARY_STOP_LOSS_ROI_PCT = 3.80
TRAILING_SL_ENABLED = True
TRAILING_ACTIVATION_PCT = 7.25
TRAILING_INITIAL_LOCK_ROI_PCT = 5.30
TRAILING_RETRACEMENT_PCT = 20.0
PROTECTION_VERIFY_SECONDS = 15
PROTECTION_RETRY_SECONDS = 5
POSITION_RECOVERY_INTERVAL = 10
LIVE_POSITION_SYNC_INTERVAL = 6.0   # Get Positions limit is 20/60s -> 10/min here, rest is for fills/watchdog
LIVE_ORDER_POLL_SECONDS = 0.75
LIVE_ORDER_POLL_ATTEMPTS = 12
RECOVERY_LOCK = False
LIVE_POSITION_SYNC_THREAD = None
PROTECTION_WATCHDOG_THREAD = None

PRINT_INTERVAL = 5.0

# ============================================================
# BACKGROUND OUTPUT ROUTING
#
# Keep the dashboard/keyboard clean on the terminal.
# Main thread + dashboard + keyboard remain visible.
# Other worker-thread prints are written to a persistent log.
# ============================================================

BACKGROUND_LOG_FILE = ".coinswitch_v5_background.log"
VISIBLE_OUTPUT_THREADS = {
    "MainThread",
    "dashboard-keyboard",
    "display_loop",
}

_original_print = builtins.print
_background_print_lock = threading.RLock()


def print(*args, **kwargs):
    """
    Route background-thread console output to a log file while
    keeping dashboard and keyboard output visible on terminal.
    """
    try:
        thread_name = threading.current_thread().name

        if thread_name in VISIBLE_OUTPUT_THREADS:
            return _original_print(*args, **kwargs)

        timestamp = datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )

        try:
            rendered = " ".join(str(arg) for arg in args)
        except Exception:
            rendered = repr(args)

        line = f"[{timestamp}] [{thread_name}] {rendered}"

        # Respect simple newline/end usage while keeping the log readable.
        end = kwargs.get("end", "\n")

        with _background_print_lock:
            with open(
                BACKGROUND_LOG_FILE,
                "a",
                encoding="utf-8",
            ) as log:
                log.write(line + end)
                log.flush()

    except Exception:
        # Output routing must NEVER break trading logic.
        try:
            _original_print(*args, **kwargs)
        except Exception:
            pass



# Dashboard box width = the full terminal width (auto-detected), so it
# fills the whole phone screen (e.g. Termux) edge to edge.

# --------------------------------------------------------------
# WATCHLIST (built from the HFT scanner snapshot - see
# watchlist_refresh_loop() / load_hft_watchlist())
# --------------------------------------------------------------

# How often (seconds) the watchlist is re-scanned.
WATCHLIST_REFRESH_SECONDS = 4 * 60 * 60   # 4 hours


# --------------------------------------------------------------
# CAPITAL ALLOCATION
# --------------------------------------------------------------
# The exchange-reported/effective account capital is the source of
# truth in LIVE_MODE. Only MAX_ALLOCATION_PCT of that capital may be
# deployed as position margin.
#
# Each NEW entry may use at most ENTRY_ALLOCATION_PCT of the
# MAX_ALLOCATION_PCT deployment budget as margin. Example: $1,000
# account -> $500 maximum deployment and at most $50 margin per entry.
# At 10x leverage that $50 margin corresponds to up to $500 notional.
#
# The allocation limits are applied to MARGIN, not full notional.
# --------------------------------------------------------------

TOTAL_CAPITAL = 10000.0        # fallback / initial paper capital
MAX_ALLOCATION_PCT = 50.0      # maximum total MARGIN deployment
ENTRY_ALLOCATION_PCT = 10.0    # maximum MARGIN allocation per new entry

# Paper-mode fee rate per executed side, applied to notional.
# Live mode never uses this value; it records exchange-reported fees.
PAPER_FEE_RATE = 0.0005
PAPER_FEE_GST_RATE = 0.18

# C = LOW CAPITAL MODE: remove 50% deployment cap; use min_qty
# and select the highest affordable leverage up to 40x.
LOW_CAPITAL_MODE = False
LOW_CAPITAL_LEVERAGES = (10, 20, 40)

# M = MAX LEVERAGE MODE (NORMAL MODE ONLY)
# ON  -> NORMAL entries use each coin's exchange max leverage
# OFF -> NORMAL entries use configured TRADE_LEVERAGE (10x)
# C/LOW CAPITAL MODE ignores M.
MAX_LEVERAGE_MODE = False

# HFT scanner snapshot produced by hft_final_watchlist_v5.py
HFT_WATCHLIST_FILE = "hft_final_watchlist_v5.json"
HFT_REFRESH_SECONDS = 30.0
HFT_MIN_QUALITY = {"CONFIRMED", "WATCH"}
HFT_ALLOW_WATCH_FALLBACK = True

# Funding protection: no NEW entry inside this blackout window.
# CoinSwitch perpetual funding is treated as a 4-hour cycle here.
# Set these UTC times to the exchange's actual funding schedule if it differs.
FUNDING_SCHEDULE_UTC = [(0, 0), (4, 0), (8, 0), (12, 0), (16, 0), (20, 0)]
FUNDING_BLACKOUT_MINUTES = 5

# Account balance refresh. The execution API balance is the source of truth
# for capital used/available calculations; the old hardcoded 10000 is fallback only.
BALANCE_REFRESH_SECONDS = 30.0

# --------------------------------------------------------------
# API LIMITS  (CoinSwitch docs, "Errors & Rate Limits", futures,
# applied PER API KEY - so the HFT scanner and this bot share them if
# they use the same key).   (max requests, window seconds)
# --------------------------------------------------------------
API_SAFETY = 0.85            # use at most 85% of every documented limit
API_LIMITS = {
    "order_place": (20, 60), "order_cancel": (10, 60), "order_cancel_all": (10, 60),
    "order_status": (20, 60), "orders_open": (20, 60), "orders_closed": (20, 60),
    "positions": (20, 60), "leverage_set": (10, 60), "leverage_get": (20, 60),
    "add_margin": (10, 60), "wallet": (20, 60), "transactions": (20, 60),
    "instrument": (100, 60), "order_book": (100, 60), "ticker": (100, 60),
    "all_pairs_ticker": (100, 60), "klines": (30, 60), "trades": (100, 60),
}
# A NEW entry is only started when these buckets still have this many free
# slots, so a fresh trade can never starve stop-loss / trailing / exit calls.
API_ENTRY_RESERVE = {"order_place": 6, "positions": 4, "orders_open": 3}

# --------------------------------------------------------------
# SYMBOL SCALING
# --------------------------------------------------------------
# Max symbols the bot tracks (top-ranked by scanner). The all-pairs ticker is
# ONE call for every symbol, so price load does not grow with this number;
# what grows is WebSocket streams, CPU and one-time candle seeding.
MAX_WATCHLIST_SYMBOLS = 60
# Only the top-N ranked symbols also get a WebSocket ticker stream (CPU on
# phones). All symbols are still priced every TICKER_POLL_SECONDS via REST.
WS_TICKER_TOP_N = 30
SYMBOL_PRUNE_AFTER_SEC = 3600

# Ignore a crossover if the 3 reference candles are not back-to-back (this
# happens after a WebSocket gap). Set False to restore v4 behaviour.
REQUIRE_CONTIGUOUS_CANDLES = True

# --------------------------------------------------------------
# RISK GATE (blocks NEW entries only, never exits / stops)
# --------------------------------------------------------------
RISK_GATE_ENABLED = True
RISK_DAILY_LOSS_PCT = 8.0        # rolling 24h net loss >= 8% of capital -> no new entries
RISK_MAX_CONSEC_LOSSES = 6       # this many losing trades in a row ...
RISK_PAUSE_MINUTES = 30          # ... pauses new entries for this long
RISK_MAX_SAME_SIDE = 3           # max simultaneous BUY (or SELL) positions (crypto is correlated)


# --------------------------------------------------------------
# WEBSOCKET (candles only - ticker/live price comes from REST)
# --------------------------------------------------------------

WS_URL = "wss://ws.coinswitch.co/"
NAMESPACE = "/exchange_2"
SOCKETIO_PATH = "/pro/realtime-rates-socket/futures/exchange_2"

# --------------------------------------------------------------
# REST API
# --------------------------------------------------------------

BASE_URL = "https://coinswitch.co"

# Fill these in with your own CoinSwitch PRO API key pair
# (Profile -> API Trading on CoinSwitch PRO). Both are hex strings.
API_KEY = os.environ.get("COINSWITCH_API_KEY", "")
SECRET_KEY = os.environ.get("COINSWITCH_SECRET_KEY", "")

# How often (seconds) to poll the REST all-pairs ticker for live
# prices of every symbol in the watchlist.
TICKER_POLL_SECONDS = 2.0


# ============================================================
# WEBSOCKET CLIENT
# ============================================================

sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=0,
    reconnection_delay=2,
    reconnection_delay_max=10,
)


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

ws_connected = False

# The symbols we currently want NEW entries on (refreshed every
# WATCHLIST_REFRESH_SECONDS).
watchlist = set()

# Every symbol we've ever subscribed a KLine stream for. This is a
# superset of `watchlist` - a symbol stays here (and keeps getting
# its candles/price updated) even after it drops out of the
# watchlist, for as long as it still has an open position, so an
# open trade is never abandoned mid-flight.
subscribed_symbols = set()

# Per-symbol state. Keys are symbol strings (e.g. "BTCUSDT").
# See make_symbol_state() for the shape of each value.
symbol_states = {}

# Simple trade log (kept in memory since orders are disabled).
# Each entry also carries a "symbol" key.
trade_log = []

# Persistent trading history/state. Paper and live are intentionally separate.
STATE_DIR = Path(os.environ.get("COINSWITCH_STATE_DIR", str(Path.home() / "CoinSwitch Futures" / "coinswitch_state")))
PAPER_LOG_FILE = STATE_DIR / "paper_trade_log.json"
LIVE_LOG_FILE = STATE_DIR / "live_trade_log.json"
PAPER_STATE_FILE = STATE_DIR / "paper_capital_state.json"
SESSION_STARTED_AT = time.time()

paper_capital = TOTAL_CAPITAL
paper_trade_log = []
live_trade_log = []

last_watchlist_refresh = None

# Per-symbol trading rules from Get Instrument Info: max_leverage
# and min_base_quantity. Refreshed alongside the watchlist.
instrument_info = {}
instrument_info_fetched_at = 0.0

# Symbols for which we've successfully set TRADE_LEVERAGE (only
# relevant when ORDERS_ENABLED is True - leverage must be set
# before the first order on a symbol).
leverage_set_symbols = set()
configured_leverage_by_symbol = {}

# HFT scanner state. The scanner NEVER creates an entry signal. It only
# confirms/prioritizes an entry generated by the existing 5m crossover.
hft_priority = {}
hft_loaded_at = None
hft_file_mtime = None
entry_priority_symbol = None
entry_priority_score = None

# Account/equity cache used for margin accounting and dashboard display.
account_balance = None
account_available_balance = None
account_balance_updated_at = None

# Per-symbol extras from the REST all-pairs ticker (funding, open interest ...)
ticker_extra = {}
# Symbols that currently have a WebSocket ticker stream (re-subscribed on reconnect)
ws_ticker_symbols = set()
symbol_rank = {}
_out_of_watchlist_since = {}


# ============================================================
# HELPERS
# ============================================================

def to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def to_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def utc_string(timestamp_ms):
    if timestamp_ms is None:
        return "N/A"

    try:
        dt = datetime.fromtimestamp(
            timestamp_ms / 1000.0,
            tz=timezone.utc,
        )

        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    except Exception:
        return "N/A"


def now_string():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def make_symbol_state():
    """
    Fresh per-symbol state dict. Every symbol we track gets one
    of these (candle history, signal search state, position).
    """

    return {
        "current_candle": None,
        "previous_closed_candle": None,
        "closed_candle_history": [],

        "live_price": None,
        "previous_live_price": None,

        "signal": "WAIT",
        "entry_price": None,
        "stop_loss": None,
        "signal_candle_start": None,
        "signal_triggered": False,

        "in_position": False,
        "position_closing": False,   # True while a close order is in flight
        "position_side": None,       # "BUY" or "SELL"
        "position_entry": None,
        "position_sl": None,
        "position_size": None,       # USDT notional (qty * entry price)
        "position_qty": None,        # base-asset quantity (min_qty for the symbol)
        "position_open_time": None,
        "position_margin": None,
        "capital_used_pct": None,
        "scanner_score": None,
        "scanner_quality": None,
        "scanner_bias": None,
        "entry_fee": 0.0,
        "entry_fee_base": 0.0,
        "entry_fee_gst": 0.0,

        # LIVE exchange protection / reconciliation
        "exchange_order_id": None,
        "actual_entry": None,
        "actual_qty": None,
        "actual_leverage": None,
        "strategy_sl": None,
        "emergency_sl": None,
        "active_sl": None,
        "sl_order_id": None,
        "sl_status": "NONE",
        "protection_type": "NONE",
        "protection_verified_at": None,
        "protection_last_check": None,
        "recovery_status": "NORMAL",
        "mark_price": None,
        "exchange_unrealised_pnl": None,
        "liquidation_price": None,
        "sl_distance_pct": None,
        "sl_capital_risk_pct": None,

        # TRAILING STOP state
        "trailing_active": False,
        "trailing_armed": False,
        "trailing_stage": "NONE",
        "trailing_peak": None,
        "trailing_sl": None,
        "trailing_last_order_id": None,

        # AI model snapshot taken at entry (used to learn at close)
        "ai_features": None,
        "ai_prob": None,
    }


def _remember_closed_candle(state, candle, limit=10):
    """Keep a small, de-duplicated history of completed candles."""
    if not isinstance(candle, dict) or candle.get("start_time") is None:
        return
    history = state.setdefault("closed_candle_history", [])
    try:
        start = int(candle["start_time"])
    except (TypeError, ValueError):
        return
    cleaned = [c for c in history if isinstance(c, dict) and c.get("start_time") != start]
    cleaned.append({
        "start_time": start,
        "open": to_float(candle.get("open")),
        "high": to_float(candle.get("high")),
        "low": to_float(candle.get("low")),
        "close": to_float(candle.get("close")),
        "volume": to_float(candle.get("volume")),
    })
    cleaned.sort(key=lambda c: int(c.get("start_time") or 0))
    state["closed_candle_history"] = cleaned[-limit:]


def _last_closed_candles(state, count=3):
    """Return the most recent completed candles, including previous_closed_candle."""
    history = list(state.get("closed_candle_history") or [])
    previous = state.get("previous_closed_candle")
    if isinstance(previous, dict):
        pstart = previous.get("start_time")
        if not any(c.get("start_time") == pstart for c in history if isinstance(c, dict)):
            history.append(dict(previous))
    unique = {}
    for candle in history:
        if isinstance(candle, dict) and candle.get("start_time") is not None:
            unique[candle.get("start_time")] = candle
    ordered = sorted(unique.values(), key=lambda c: int(c.get("start_time") or 0))
    return ordered[-count:]


def ensure_symbol_state(symbol):
    """
    Must be called under state_lock. Creates a fresh state entry
    for `symbol` if one doesn't already exist.
    """

    if symbol not in symbol_states:
        symbol_states[symbol] = make_symbol_state()


# ============================================================
# PERSISTENT PAPER/LIVE TRADE STATE
# ============================================================

def _load_json_file(path, default):
    try:
        if not path.exists():
            return default
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data is not None else default
    except Exception as e:
        print(f"[Persistence] Could not load {path}: {e}")
        return default


def _atomic_write_json(path, data):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"[Persistence] Could not save {path}: {e}")
        return False


def load_persistent_state():
    """Restore paper capital and separate paper/live histories on startup."""
    global paper_capital, paper_trade_log, live_trade_log, trade_log

    paper_trade_log = _load_json_file(PAPER_LOG_FILE, [])
    live_trade_log = _load_json_file(LIVE_LOG_FILE, [])
    saved_capital = _load_json_file(PAPER_STATE_FILE, {})

    if isinstance(saved_capital, dict) and to_float(saved_capital.get("paper_capital")) is not None:
        paper_capital = float(saved_capital["paper_capital"])
    else:
        paper_capital = TOTAL_CAPITAL

    # Keep the existing in-memory trade_log API used by the strategy.
    trade_log = []

    print(f"[Persistence] Paper capital: {paper_capital:.6f}")
    print(f"[Persistence] Paper history: {len(paper_trade_log)} trades")
    print(f"[Persistence] Live history : {len(live_trade_log)} trades")


def save_paper_capital():
    _atomic_write_json(PAPER_STATE_FILE, {
        "paper_capital": paper_capital,
        "updated_at": time.time(),
    })


def _paper_fee_for_notional(notional):
    """Paper taker fee plus 18% GST, for one executed side's notional."""
    try:
        amount = abs(float(notional or 0.0))
    except (TypeError, ValueError):
        amount = 0.0
    base_fee = amount * PAPER_FEE_RATE
    gst = base_fee * PAPER_FEE_GST_RATE
    return base_fee + gst, base_fee, gst


def _fee_number(value):
    value = to_float(value)
    return abs(value) if value is not None else None


def extract_actual_fee(payload):
    """Extract an actual fee/commission amount when the exchange returns it in the order payload."""
    if payload is None:
        return None
    keys = {
        "fee", "fees", "trading_fee", "transaction_fee", "commission",
        "commission_amount", "fee_amount", "executed_fee", "total_fee"
    }
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in keys:
                if isinstance(v, dict):
                    for sub in ("amount", "value", "total", "fee", "commission"):
                        n = _fee_number(v.get(sub))
                        if n is not None:
                            return n
                else:
                    n = _fee_number(v)
                    if n is not None:
                        return n
        for v in payload.values():
            n = extract_actual_fee(v)
            if n is not None:
                return n
    elif isinstance(payload, list):
        for item in payload:
            n = extract_actual_fee(item)
            if n is not None:
                return n
    return None


def extract_actual_execution_price(payload):
    """Extract the exchange-reported average execution/fill price."""
    if payload is None:
        return None

    keys = {
        "avg_execution_price",
        "average_execution_price",
        "avg_fill_price",
        "average_fill_price",
        "execution_price",
        "fill_price",
    }

    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in keys:
                n = to_float(v)
                if n is not None and n > 0:
                    return n

        for v in payload.values():
            n = extract_actual_execution_price(v)
            if n is not None:
                return n

    elif isinstance(payload, list):
        for item in payload:
            n = extract_actual_execution_price(item)
            if n is not None:
                return n

    return None


def rolling_24h_records(history):
    cutoff = time.time() - 24 * 60 * 60
    return [r for r in history if to_float(r.get("closed_at")) is not None and float(r["closed_at"]) >= cutoff]


def rolling_24h_metrics(history):
    rows = rolling_24h_records(history)
    gross = sum(float(r.get("pnl_usdt") or 0.0) for r in rows)
    fees = sum(float(r.get("fee") or 0.0) for r in rows)
    net = gross - fees
    return rows, gross, fees, net


# ============================================================
# REST API (Ed25519-signed requests)
# ============================================================

def sign_request(method, path, params=None):
    """
    Build the headers and final URL path for an authenticated
    CoinSwitch request (Ed25519 signing), per CoinSwitch PRO's
    API Trading docs.
    """

    method = method.upper()

    if params:
        sep = "&" if "?" in path else "?"
        path = path + sep + urllib.parse.urlencode(params)

    decoded_path = urllib.parse.unquote_plus(path)

    epoch = str(int(time.time() * 1000))

    message = method + decoded_path + epoch

    secret = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(SECRET_KEY)
    )

    signature = secret.sign(message.encode("utf-8")).hex()

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": signature,
        "X-AUTH-EPOCH": epoch,
    }

    return headers, decoded_path



# ============================================================
# API RATE-LIMIT GUARD
#
# Every REST call in this file goes through cs_get / cs_post / cs_delete.
# Each call is mapped to a CoinSwitch endpoint bucket and must obtain a slot
# in that bucket's sliding window BEFORE it is sent. On HTTP 429 the bucket is
# paused with exponential back-off + jitter (per CoinSwitch's Backoff guide).
# ============================================================
import collections
import queue as _queue
import random as _rl_random
import uuid


class LocalRateLimited(requests.exceptions.RequestException):
    """Raised when the local guard could not get a slot in time (nothing was sent)."""


_api_lock = threading.Lock()
_api_hits = {}
_api_blocked_until = {}
_api_429_streak = {}
_api_stats = {}

_API_MAX_WAIT_DEFAULT = 12.0
_API_MAX_WAIT = {
    "order_place": 25.0, "order_cancel": 25.0, "order_cancel_all": 25.0,
    "order_status": 20.0, "positions": 20.0, "orders_open": 20.0,
    "orders_closed": 20.0, "leverage_set": 20.0, "wallet": 15.0,
}


def _api_bucket(method, path):
    p = path.split("?", 1)[0].rstrip("/")
    m = method.upper()
    if p.endswith("/futures/order"):
        return {"POST": "order_place", "DELETE": "order_cancel"}.get(m, "order_status")
    if p.endswith("/futures/orders/open"):
        return "orders_open"
    if p.endswith("/futures/orders/closed"):
        return "orders_closed"
    if p.endswith("/futures/cancel_all"):
        return "order_cancel_all"
    if p.endswith("/futures/leverage"):
        return "leverage_set" if m == "POST" else "leverage_get"
    if p.endswith("/futures/add_margin"):
        return "add_margin"
    if p.endswith("/futures/wallet_balance"):
        return "wallet"
    if p.endswith("/futures/transactions"):
        return "transactions"
    if p.endswith("/futures/instrument_info"):
        return "instrument"
    if p.endswith("/futures/order_book"):
        return "order_book"
    if p.endswith("/futures/all-pairs/ticker"):
        return "all_pairs_ticker"
    if p.endswith("/futures/ticker"):
        return "ticker"
    if p.endswith("/futures/klines"):
        return "klines"
    if p.endswith("/futures/trades"):
        return "trades"
    if p.endswith("/futures/positions"):
        return "positions"
    return "other"


def _api_cap(bucket):
    limit, window = API_LIMITS.get(bucket, (60, 60))
    return max(1, int(limit * API_SAFETY)), float(window)


def api_free(bucket):
    """How many more calls the bucket can take right now."""
    cap, window = _api_cap(bucket)
    now = time.time()
    with _api_lock:
        dq = _api_hits.get(bucket)
        if _api_blocked_until.get(bucket, 0.0) > now:
            return 0
        if not dq:
            return cap
        while dq and now - dq[0] >= window:
            dq.popleft()
        return max(0, cap - len(dq))


def _api_acquire(bucket, max_wait):
    """Block until the bucket has a free slot. Returns seconds waited, or None."""
    cap, window = _api_cap(bucket)
    start = time.time()
    while True:
        now = time.time()
        with _api_lock:
            dq = _api_hits.setdefault(bucket, collections.deque())
            while dq and now - dq[0] >= window:
                dq.popleft()
            blocked = _api_blocked_until.get(bucket, 0.0)
            if blocked <= now and len(dq) < cap:
                dq.append(now)
                st = _api_stats.setdefault(bucket, {"calls": 0, "r429": 0, "waited": 0.0})
                st["calls"] += 1
                st["waited"] += now - start
                return now - start
            wait = max(
                blocked - now,
                (dq[0] + window - now) if len(dq) >= cap else 0.0,
                0.05,
            )
        if (time.time() - start) + min(wait, 1.0) > max_wait:
            return None
        time.sleep(min(wait + 0.02, 1.0))


def _api_note_429(bucket, response):
    _, window = _api_cap(bucket)
    retry_after = None
    try:
        retry_after = float(response.headers.get("Retry-After"))
    except Exception:
        retry_after = None
    with _api_lock:
        streak = _api_429_streak.get(bucket, 0) + 1
        _api_429_streak[bucket] = streak
        base = retry_after if retry_after and retry_after > 0 else min(window, 15.0 * (2 ** (streak - 1)))
        delay = min(60.0, base) * (1.0 + _rl_random.uniform(-0.25, 0.25))
        _api_blocked_until[bucket] = time.time() + delay
        st = _api_stats.setdefault(bucket, {"calls": 0, "r429": 0, "waited": 0.0})
        st["r429"] += 1
    print(f"[API 429] bucket={bucket} streak={streak} paused {delay:.0f}s")


def cs_request(method, url, **kwargs):
    path = url.replace(BASE_URL, "", 1)
    bucket = _api_bucket(method, path)
    waited = _api_acquire(bucket, _API_MAX_WAIT.get(bucket, _API_MAX_WAIT_DEFAULT))
    if waited is None:
        raise LocalRateLimited(f"local rate-limit guard: '{bucket}' budget exhausted")

    # The signature carries a timestamp (CoinSwitch rejects skew > 60s).
    # If we had to wait, sign again so a queued call can never go out stale.
    if waited > 2.0 and API_KEY and SECRET_KEY:
        try:
            fresh, _ = sign_request(method, path)
            kwargs["headers"] = {**(kwargs.get("headers") or {}), **fresh}
        except Exception:
            pass

    response = requests.request(method, url, **kwargs)
    if response.status_code == 429:
        _api_note_429(bucket, response)
    elif response.status_code < 500:
        with _api_lock:
            _api_429_streak[bucket] = 0
    return response


def cs_get(url, **kwargs):
    return cs_request("GET", url, **kwargs)


def cs_post(url, **kwargs):
    return cs_request("POST", url, **kwargs)


def cs_delete(url, **kwargs):
    return cs_request("DELETE", url, **kwargs)


def api_entry_budget_ok():
    for bucket, need in API_ENTRY_RESERVE.items():
        if api_free(bucket) < need:
            return False, bucket
    return True, None


def api_usage_snapshot():
    now = time.time()
    rows = []
    with _api_lock:
        for bucket, (limit, window) in API_LIMITS.items():
            dq = _api_hits.get(bucket)
            if not dq:
                continue
            used = sum(1 for t in dq if now - t < window)
            st = _api_stats.get(bucket, {})
            rows.append((bucket, used, limit, st.get("calls", 0), st.get("r429", 0), st.get("waited", 0.0)))
    return rows


def api_usage_lines():
    rows = api_usage_snapshot()
    if not rows:
        return ["API Load  : idle"]
    worst = max(rows, key=lambda r: r[1] / max(1, r[2]))
    total_429 = sum(r[4] for r in rows)
    total_calls = sum(r[3] for r in rows)
    return [
        f"API Load  : peak {worst[0]} {worst[1]}/{worst[2]} per 60s | "
        f"calls {total_calls} | 429s {total_429}"
    ]


def api_monitor_loop():
    while True:
        time.sleep(60)
        try:
            for bucket, used, limit, calls, r429, waited in api_usage_snapshot():
                if used >= 0.6 * limit or r429:
                    print(f"[API LOAD] {bucket}: {used}/{limit} per 60s | total calls {calls} | 429s {r429} | waited {waited:.1f}s")
        except Exception as exc:
            print(f"[API LOAD] monitor error: {exc}")


_api_monitor_started = False


def start_api_monitor():
    global _api_monitor_started
    if _api_monitor_started:
        return
    _api_monitor_started = True
    threading.Thread(target=api_monitor_loop, daemon=True, name="api-monitor").start()


def fetch_all_pairs_ticker():
    """
    GET /trade/api/v2/futures/all-pairs/ticker

    Returns a dict: { "BTCUSDT": {"price": .., "bid": .., "ask": ..,
    "mark": .., "timestamp": .., "pct24h": ..}, ... } on success,
    or None on any failure. Never raises.

    CoinSwitch can intermittently return HTTP 401 "Invalid access" even
    when freshly signed requests immediately before/after succeed.
    Therefore only HTTP 401 gets ONE fresh-signature retry.
    No retry is performed for 429 or other errors.
    """

    try:

        response = None

        for attempt in range(2):

            headers, path = sign_request(
                "GET",
                "/trade/api/v2/futures/all-pairs/ticker",
                params={"exchange": "EXCHANGE_2"},
            )

            if attempt == 1:
                time.sleep(0.35)

            response = cs_get(
                BASE_URL + path,
                headers=headers,
                timeout=10,
            )

            if response.status_code != 401:
                break

            body_preview = response.text[:300]

            if attempt == 0:
                print(
                    f"[REST All-Pairs Ticker 401] "
                    f"Transient auth failure; retrying once. "
                    f"BODY: {body_preview}"
                )
            else:
                print(
                    f"[REST All-Pairs Ticker 401] "
                    f"Retry also failed. BODY: {body_preview}"
                )

        response.raise_for_status()

        payload = response.json()

        raw = payload.get("data")

        if not isinstance(raw, dict):
            return None

        result = {}

        for symbol, ticker in raw.items():

            if not isinstance(ticker, dict):
                continue

            price = to_float(ticker.get("last_price"))

            if price is None:
                continue

            result[symbol] = {
                "price": price,
                "bid": to_float(ticker.get("best_bid_price")),
                "ask": to_float(ticker.get("best_ask_price")),
                "mark": to_float(ticker.get("mark_price")),
                "timestamp": to_int(ticker.get("timestamp")),
                "pct24h": to_float(ticker.get("price_24h_pcnt")),
                "funding": to_float(ticker.get("funding_rate")),
                "next_funding": to_int(ticker.get("next_funding_timestamp")),
                "oi_value": to_float(ticker.get("open_interest_value")),
                "quote_vol24": to_float(ticker.get("quote_asset_volume_24h")),
            }

        return result

    except Exception as e:

        print(f"[REST All-Pairs Ticker Error] {e}")
        return None


def fetch_instrument_info():
    """
    GET /trade/api/v2/futures/instrument_info

    Returns a dict: { "BTCUSDT": {"max_leverage": .., "min_qty": ..}, ... }
    on success, or None on any failure. Never raises.
    """

    try:

        headers, path = sign_request(
            "GET",
            "/trade/api/v2/futures/instrument_info",
            params={"exchange": "EXCHANGE_2"},
        )

        response = cs_get(
            BASE_URL + path,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()

        raw = payload.get("data")

        if not isinstance(raw, dict):
            return None

        result = {}

        for symbol, info in raw.items():

            if not isinstance(info, dict):
                continue

            max_leverage = to_float(info.get("max_leverage"))
            min_qty = to_float(info.get("min_base_quantity"))

            if max_leverage is None or min_qty is None:
                continue

            result[symbol] = {
                "max_leverage": max_leverage,
                "min_qty": min_qty,
            }

        return result

    except Exception as e:

        print(f"[REST Instrument Info Error] {e}")
        return None


# ============================================================
# REAL ORDER PLACEMENT (only actually called when ORDERS_ENABLED)
# ============================================================


def set_symbol_leverage(symbol, leverage=None):
    """
    Set the requested leverage for a symbol.
    Normal mode defaults to TRADE_LEVERAGE (10x).
    LOW CAPITAL MODE uses the selected 10x/20x/40x candidate.
    """
    try:
        selected = int(float(
            leverage if leverage is not None else TRADE_LEVERAGE
        ))

        if selected <= 0:
            return False

        headers, path = sign_request(
            "POST", "/trade/api/v2/futures/leverage"
        )

        body = {
            "symbol": symbol,
            "exchange": "EXCHANGE_2",
            "leverage": selected,
        }

        response = cs_post(
            BASE_URL + path,
            headers=headers,
            json=body,
            timeout=10,
        )

        response.raise_for_status()

        leverage_set_symbols.add(symbol)
        configured_leverage_by_symbol[symbol] = selected

        print(f"[Leverage] {symbol} set to {selected}x")
        return True

    except Exception as e:
        print(f"[Leverage Error] ({symbol}) {e}")
        return False

def place_market_order(symbol, side, quantity, reduce_only=False):
    """
    POST /trade/api/v2/futures/order (MARKET order).

    side        — "BUY" or "SELL"
    quantity    — base-asset quantity
    reduce_only — True when this order is meant to CLOSE an
                  existing position (never opens a new/opposite one)

    Returns the parsed response dict on success, or None on any
    failure. Never raises.
    """

    try:

        headers, path = sign_request(
            "POST", "/trade/api/v2/futures/order"
        )

        body = {
            "exchange": "EXCHANGE_2",
            "symbol": symbol,
            "side": side,
            "order_type": "MARKET",
            "quantity": quantity,
            # CoinSwitch de-duplicates on client_order_id, so reconciliation
            # after a timeout / 5xx can never create a second position.
            "client_order_id": str(uuid.uuid4()),
        }

        if reduce_only:
            body["reduce_only"] = True

        response = cs_post(
            BASE_URL + path,
            headers=headers,
            json=body,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        print(f"[Order] {symbol} {side} qty={quantity} reduce_only={reduce_only} -> {data}")

        return data

    except (requests.Timeout, requests.ConnectionError) as e:

        # A timeout/connection drop after submission is NOT equivalent to "order failed".
        # The exchange may have accepted/executed the order. Never resend
        # automatically. Reconciliation must determine the real state.
        print(f"[Order AMBIGUOUS] ({symbol} {side} qty={quantity}) network error: {e}")
        return {"_ambiguous": True, "_error": "timeout"}

    except Exception as e:

        # CoinSwitch docs: a 5xx is NOT a failed order - the outcome is unknown.
        _resp5 = locals().get("response")
        if _resp5 is not None and getattr(_resp5, "status_code", 0) >= 500:
            print(
                f"[Order AMBIGUOUS] ({symbol} {side} qty={quantity}) "
                f"HTTP {_resp5.status_code}; outcome unknown, reconciling."
            )
            return {"_ambiguous": True, "_error": f"http_{_resp5.status_code}"}

        print(f"[Order Error] ({symbol} {side} qty={quantity}) {e}")
        if "response" in locals():
            print(f"[Order HTTP] status={response.status_code}")
            print(f"[Order Response] {response.text}")
        return None


# ============================================================
# HISTORICAL KLINES (REST) - used to seed a newly-added symbol's
# "previous closed candle" immediately, instead of waiting for
# it to arrive organically over the live WebSocket stream.
# ============================================================

def fetch_recent_klines(symbol, limit=3):
    """
    GET /trade/api/v2/futures/klines

    Returns a list of candle dicts (sorted oldest -> newest), each
    with: start_time, close_time, open, high, low, close, volume.
    Returns None on any failure. Never raises.
    """

    try:

        headers, path = sign_request(
            "GET",
            "/trade/api/v2/futures/klines",
            params={
                "symbol": symbol.lower(),
                "exchange": "EXCHANGE_2",
                "interval": INTERVAL,
                "limit": limit,
            },
        )

        response = cs_get(
            BASE_URL + path,
            headers=headers,
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()

        raw = payload.get("data")

        if not isinstance(raw, list):
            return None

        candles = []

        for row in raw:

            start_time = to_int(row.get("start_time"))
            close_time = to_int(row.get("close_time"))
            o = to_float(row.get("o"))
            h = to_float(row.get("h"))
            l = to_float(row.get("l"))
            c = to_float(row.get("c"))
            v = to_float(row.get("volume"))

            if start_time is None or o is None or h is None or l is None or c is None:
                continue

            candles.append(
                {
                    "start_time": start_time,
                    "close_time": close_time,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )

        candles.sort(key=lambda x: x["start_time"])

        return candles

    except Exception as e:

        print(f"[REST Klines Error] ({symbol}) {e}")
        return None


def seed_symbol_history(symbol):
    """
    Fetches the last few candles for `symbol` via REST and, if the
    symbol doesn't already have candle data (from the live
    WebSocket stream), seeds its previous_closed_candle (and, if
    available, current_candle too) so the strategy can evaluate
    signals immediately instead of waiting 5-10 minutes for two
    live candles to pass.
    """

    candles = fetch_recent_klines(symbol, limit=6)

    if not candles:
        return

    now_ms = int(time.time() * 1000)

    # A candle counts as fully closed only if its close_time has
    # actually passed - never trust the last row blindly, since
    # some APIs include the still-forming candle as the last row.
    closed = [
        c for c in candles
        if c["close_time"] is not None and c["close_time"] <= now_ms
    ]

    if not closed:
        return

    previous = closed[-1]

    # Whatever candle (if any) starts after `previous` is the
    # current, still-forming candle.
    current_candidate = None

    for c in candles:
        if c["start_time"] > previous["start_time"]:
            current_candidate = c
            break

    def strip(c):
        return {
            "start_time": c["start_time"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
        }

    with state_lock:

        if symbol not in symbol_states:
            return

        state = symbol_states[symbol]

        # Only seed if the live WebSocket hasn't already populated
        # this - never overwrite live data with a stale REST call.
        for closed_candle in closed[-5:]:
            _remember_closed_candle(state, strip(closed_candle))

        if state["previous_closed_candle"] is None:
            state["previous_closed_candle"] = strip(previous)

        if state["current_candle"] is None and current_candidate is not None:
            state["current_candle"] = strip(current_candidate)

    print(
        f"[Seed] {symbol}: previous candle loaded from REST "
        f"(O:{previous['open']:.6f} H:{previous['high']:.6f} "
        f"L:{previous['low']:.6f} C:{previous['close']:.6f})"
    )


# ============================================================
# WATCHLIST (Top Gainers / Top Losers)
# ============================================================

def fetch_futures_balance():
    """
    Fetch the actual CoinSwitch Futures USDT wallet balance.

    Returns:
        {
            "total": total USDT balance,
            "available": available USDT balance,
        }

    The Futures wallet endpoint is:
        GET /trade/api/v2/futures/wallet_balance
    """

    try:
        headers, path = sign_request(
            "GET",
            "/trade/api/v2/futures/wallet_balance",
        )

        response = cs_get(
            BASE_URL + path,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data")

        if not isinstance(data, dict):
            print("[Futures Balance Error] Invalid wallet response")
            return None

        base_asset_balances = data.get("base_asset_balances")

        if not isinstance(base_asset_balances, list):
            print("[Futures Balance Error] base_asset_balances not found")
            return None

        for item in base_asset_balances:
            if not isinstance(item, dict):
                continue

            if str(item.get("base_asset", "")).upper() != "USDT":
                continue

            balances = item.get("balances")

            if not isinstance(balances, dict):
                print("[Futures Balance Error] USDT balances not found")
                return None

            total = to_float(
                balances.get("total_balance")
            )

            available = to_float(
                balances.get("total_available_balance")
            )

            if total is None:
                print("[Futures Balance Error] total_balance not found")
                return None

            if available is None:
                print(
                    "[Futures Balance Error] "
                    "total_available_balance not found"
                )
                return None

            blocked = to_float(
                balances.get("total_blocked_balance")
            )
            position_margin = to_float(
                balances.get("total_position_margin")
            )
            open_order_margin = to_float(
                balances.get("total_open_order_margin")
            )

            print(
                f"[Futures Balance] "
                f"USDT total={total:.6f} "
                f"available={available:.6f} "
                f"blocked={blocked if blocked is not None else 0.0:.6f} "
                f"position_margin={position_margin if position_margin is not None else 0.0:.6f} "
                f"open_order_margin={open_order_margin if open_order_margin is not None else 0.0:.6f}"
            )

            return {
                "total": total,
                "available": available,
            }

        print("[Futures Balance Error] USDT wallet row not found")
        return None

    except Exception as e:
        print(f"[Futures Balance Error] {e}")

        if "response" in locals():
            print(f"[Futures Balance HTTP] status={response.status_code}")
            print(f"[Futures Balance Response] {response.text}")

        return None

def refresh_account_balance():
    global account_balance, account_available_balance, account_balance_updated_at
    data = fetch_futures_balance()
    if data is None:
        return False
    with state_lock:
        account_balance = data["total"]
        account_available_balance = data["available"]
        account_balance_updated_at = time.time()
    return True


def effective_total_capital():
    if LIVE_MODE:
        value = account_balance
        return value if value is not None and value > 0 else 0.0

    if PAPER_MODE:
        return paper_capital

    return 0.0


def effective_available_capital():
    if LIVE_MODE:
        value = account_available_balance
        if value is not None and value >= 0:
            return value
        return effective_total_capital()

    if PAPER_MODE:
        return paper_capital

    return 0.0



def max_deployable_margin():
    capital = effective_total_capital()
    if LOW_CAPITAL_MODE:
        return capital
    return capital * (MAX_ALLOCATION_PCT / 100.0)

def funding_status(now=None):
    """Return funding blackout status using the configured 4-hour schedule."""
    if now is None:
        now = datetime.now(timezone.utc)

    candidates = []
    for day_offset in (-1, 0, 1):
        day = now.date()
        base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        base = base + timedelta(days=day_offset)
        for hour, minute in FUNDING_SCHEDULE_UTC:
            candidates.append(base.replace(hour=hour, minute=minute, second=0, microsecond=0))

    nearest = min(candidates, key=lambda x: abs((x - now).total_seconds()))
    delta_seconds = (now - nearest).total_seconds()
    blackout_seconds = FUNDING_BLACKOUT_MINUTES * 60
    blocked = abs(delta_seconds) <= blackout_seconds
    next_funding = min((x for x in candidates if x > now), default=None)
    return blocked, nearest, next_funding, delta_seconds


def load_hft_watchlist(force=False):
    """Load the scanner snapshot. Scanner data is an entry-priority layer only."""
    global hft_priority, hft_loaded_at, hft_file_mtime
    path = Path(HFT_WATCHLIST_FILE)
    if not path.exists():
        return False

    try:
        mtime = path.stat().st_mtime
        if not force and hft_file_mtime == mtime:
            return True

        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for key in ("final_long", "final_short", "top_10_long", "top_10_short"):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(value)
        if not rows:
            rows = payload.get("all_results") or []
        new_priority = {}

        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if not symbol:
                continue

            momentum = row.get("momentum") or {}
            market = row.get("market") or {}
            orderbook = row.get("orderbook") or {}
            execution = row.get("execution") or {}

            move_5m = to_float(momentum.get("momentum_5m_pct")) or 0.0
            move_1m = to_float(momentum.get("momentum_1m_pct")) or 0.0
            score = to_float(row.get("score")) or 0.0
            quality = str(row.get("quality", "")).upper()
            if not quality:
                quality = "CONFIRMED" if row.get("confirmed") else "WATCH"

            # V5 uses direction-adjusted weighted orderbook strength in WOB.
            wob = to_float(row.get("weighted_orderbook"))
            if wob is None:
                wob = to_float(row.get("wob"))
            if wob is None:
                wob = 0.0

            bias = str(row.get("signal", "")).upper()
            if bias not in {"LONG", "SHORT"}:
                bias = "LONG" if move_5m > 0 else "SHORT" if move_5m < 0 else "MIXED"

            new_priority[symbol] = {
                "score": score,
                "quality": quality,
                "bias": bias,
                "move_5m": move_5m,
                "move_1m": move_1m,
                "wob": wob,
                "turnover_24h": to_float(market.get("turnover_24h")) or 0.0,
                "execution_score": to_float(execution.get("execution_score")) or 0.0,
            }

        with state_lock:
            hft_priority = new_priority
            hft_loaded_at = time.time()
            hft_file_mtime = mtime
        return True

    except Exception as e:
        print(f"[HFT Scanner] Load error: {e}")
        return False


# ============================================================
# AI MODEL v2 - local, pure Python (no new dependencies).
#
# What it does (it NEVER places orders and never touches SL):
#   1. Reads the market at entry time: candle shape, volatility,
#      volume, order-book spread, scanner data (+ its age), BTC
#      regime, market breadth, funding clock, weekday/hour.
#   2. Learns from BOTH real closed trades AND "shadow" samples:
#      a passive simulator that, every few minutes per symbol and
#      side, records the features and later simulates the bot's own
#      SL + trailing + fees to get the true outcome. This removes
#      the selection bias of learning only from trades it took.
#   3. Model = blend of logistic regression + boosted decision
#      stumps (win probability), plus a ridge model for expected
#      ROI after fees. Time-ordered validation (AUC / Brier),
#      Platt calibration, drift check, and live-vs-predicted
#      monitoring on real trades.
#
# Entry/veto behaviour is unchanged (same ai_should_veto contract).
# ============================================================

import atexit
import random
from bisect import bisect_left
from operator import mul as _mul

AI_LEGACY_FEATURES = 14      # v1 vectors had these 14 features
AI_V2_FEATURES = 36          # v2 vectors had these 36 features
AI_MODEL_VERSION = 3

AI_FEATURE_NAMES = (
    # --- v1 features (order must not change) ---
    "side", "scanner_score", "wob", "momentum_5m", "momentum_1m",
    "execution_score", "turnover_log", "prev_body_pct", "prev_range_pct",
    "cross_distance_pct", "spread_pct", "hour_sin", "hour_cos",
    "mins_to_funding",
    # --- v2 features (all optional: missing -> imputed neutral) ---
    "atr_pct", "range_ratio", "vol_ratio", "opp_streak",
    "rejection_wick", "against_wick", "close_pos",
    "trend_3", "trend_10", "price_vs_mean",
    "btc_trend", "btc_range_pct", "breadth",
    "scanner_age_s", "scanner_quality", "scanner_bias_match",
    "mark_dev_pct", "dow_sin", "dow_cos",
    "spread_to_atr", "sl_to_atr", "setup_match",
    # --- v3 features (all optional: missing -> imputed neutral) ---
    "funding_pct", "oi_log", "pct24h", "rs_btc",
    "sym_wr", "bot_streak", "candle_progress",
)
AI_SAMPLES_FILE = STATE_DIR / "ai_samples.json"
AI_MODEL_FILE = STATE_DIR / "ai_model.json"

_ai_lock = threading.Lock()
ai_samples = []              # dicts: key, x, y, roi, t, src
_ai_sample_keys = set()
ai_model = {
    "version": AI_MODEL_VERSION, "mean": None, "std": None,
    "lr_w": None, "lr_b": 0.0, "gbm_f0": 0.0, "gbm": [],
    "ev_w": None, "ev_b": 0.0, "ev_mu": 0.0, "ev_sd": 1.0,
    "plat_a": 1.0, "plat_b": 0.0,
    "n": 0, "wins": 0, "n_real": 0, "wins_real": 0,
    "base_rate": None, "base_rate_real": None,
    "val": None, "edge_ok": False, "trained_at": None,
    "n_setup": 0, "wins_setup": 0, "base_rate_setup": None, "veto_thr": None,
}
ai_vetoed_count = 0
_ai_train_pending = False
_ai_train_running = False
_ai_train_force = False
_ai_last_train_end = 0.0
_ai_new_since_train = 0
_ai_dirty = False
_ai_last_flush = 0.0
_ai_live_preds = []          # (prob, y) for real trades that had an AI score
_ai_shadow_pending = []
_ai_shadow_last = {}
_ai_shadow_stats = {"opened": 0, "closed": 0, "wins": 0}
_ai_shadow_started = False


# ------------------------------------------------------------
# small math helpers
# ------------------------------------------------------------

def _ai_sigmoid(v):
    v = max(-30.0, min(30.0, v))
    return 1.0 / (1.0 + math.exp(-v))


def _ai_logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1.0 - p))


def _ai_finite(v):
    f = to_float(v)
    if f is None or math.isnan(f) or math.isinf(f):
        return None
    return float(f)


def _ai_auc(scores, labels):
    n = len(scores)
    n_pos = sum(1 for y in labels if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ai_corr(a, b):
    n = len(a)
    if n < 5:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-12 or vb <= 1e-12:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def _ai_pad_x(x, d):
    """Accept v1 (14), v2 (37) or current (d) vectors; pad/clean to length d."""
    if not isinstance(x, list):
        return None
    n = len(x)
    if n == d:
        pass
    elif n in (AI_LEGACY_FEATURES, AI_V2_FEATURES) and n < d:
        x = list(x) + [None] * (d - n)
    else:
        return None
    out = [_ai_finite(v) for v in x]
    if any(v is None for v in out[:AI_LEGACY_FEATURES]):
        return None
    return out


# ------------------------------------------------------------
# market reading -> feature vector
# ------------------------------------------------------------

def _ai_market_context():
    """Market-wide context: breadth, BTC regime, scanner age."""
    with state_lock:
        green = red = total = 0
        for sym, st in symbol_states.items():
            if sym not in watchlist:
                continue
            prev = st.get("previous_closed_candle")
            if not isinstance(prev, dict):
                continue
            o = to_float(prev.get("open"))
            c = to_float(prev.get("close"))
            if o is None or c is None:
                continue
            total += 1
            if c > o:
                green += 1
            elif c < o:
                red += 1
        btc_state = symbol_states.get("BTCUSDT")
        btc_closed = (
            [dict(c) for c in _last_closed_candles(btc_state, 4)]
            if btc_state else []
        )
        loaded_at = hft_loaded_at

    ctx = {
        "green_share": (green / total) if total >= 5 else None,
        "red_share": (red / total) if total >= 5 else None,
        "btc_closed": btc_closed,
        "scanner_age": (time.time() - loaded_at) if loaded_at else None,
    }
    return ctx


def _ai_symbol_snapshot(symbol):
    with state_lock:
        st = symbol_states.get(symbol)
        if not st:
            return None
        return {
            "prev": dict(st.get("previous_closed_candle") or {}),
            "closed": [dict(c) for c in _last_closed_candles(st, 10)],
            "bid": to_float(st.get("bid_price")),
            "ask": to_float(st.get("ask_price")),
            "mark": to_float(st.get("mark_price")),
            "live": to_float(st.get("live_price")),
            "info": dict(hft_priority.get(symbol) or {}),
            "sym": symbol,
            "extra": dict(ticker_extra.get(symbol) or {}),
            "cur_start": (st.get("current_candle") or {}).get("start_time"),
        }


def _ai_candle_vals(c):
    o = to_float(c.get("open"))
    h = to_float(c.get("high"))
    l = to_float(c.get("low"))
    cl = to_float(c.get("close"))
    v = to_float(c.get("volume"))
    if not o or o <= 0 or h is None or l is None or cl is None:
        return None
    return o, h, l, cl, v


# ---------------- v3 helpers: symbol form + bot streak ----------------
_ai_sym_ema = {}        # symbol -> (ema_win_rate, count) over setup/real outcomes
_ai_recent_real = []    # latest REAL trade outcomes (1 win / 0 loss)


def _ai_symbol_wr(sym):
    if not sym:
        return None
    with _ai_lock:
        v = _ai_sym_ema.get(sym)
    if not v or v[1] < 3:
        return None
    return v[0]


def _ai_note_symbol_outcome(sym, y):
    if not sym:
        return
    with _ai_lock:
        ema, n = _ai_sym_ema.get(sym, (None, 0))
        ema = float(y) if ema is None else 0.8 * ema + 0.2 * float(y)
        _ai_sym_ema[sym] = (ema, n + 1)


def _ai_bot_streak():
    with _ai_lock:
        recent = list(_ai_recent_real[-8:])
    if not recent:
        return None
    streak = 0
    for y in reversed(recent):
        if y:
            break
        streak += 1
    return min(streak, 6) / 6.0


def _ai_features_from(snap, side, price, ctx, now=None):
    price = _ai_finite(price)
    if price is None or price <= 0 or side not in ("BUY", "SELL"):
        return None
    if not snap:
        return None

    sign = 1.0 if side == "BUY" else -1.0
    prev = snap["prev"]
    pv = _ai_candle_vals(prev) if prev else None
    if pv is None:
        return None
    po, ph, pl, pc, _pvol = pv

    now = now or datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0
    try:
        _, _, next_funding, _ = funding_status(now)
        mins = (
            (next_funding - now).total_seconds() / 60.0
            if next_funding else 240.0
        )
    except Exception:
        mins = 240.0
    mins = max(0.0, min(240.0, mins))

    bid, ask = snap.get("bid"), snap.get("ask")
    spread = 0.0
    if bid and ask and ask >= bid:
        spread = (ask - bid) / price * 100.0

    info = snap.get("info") or {}
    prev_range_pct = (ph - pl) / po * 100.0
    cross_dist = (price - po) / po * 100.0 * sign

    base = [
        sign,
        to_float(info.get("score")) or 0.0,
        to_float(info.get("wob")) or 0.0,
        (to_float(info.get("move_5m")) or 0.0) * sign,
        (to_float(info.get("move_1m")) or 0.0) * sign,
        to_float(info.get("execution_score")) or 0.0,
        math.log10(1.0 + max(0.0, to_float(info.get("turnover_24h")) or 0.0)),
        (pc - po) / po * 100.0 * sign,
        prev_range_pct,
        cross_dist,
        spread,
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        mins / 240.0,
    ]
    if any(_ai_finite(v) is None for v in base):
        return None

    # ---------------- v2 features (optional) ----------------
    candles = []
    for c in snap.get("closed") or []:
        cv = _ai_candle_vals(c)
        if cv is not None:
            candles.append(cv)

    ranges = [(h - l) / o * 100.0 for o, h, l, _c, _v in candles]
    atr_pct = (sum(ranges) / len(ranges)) if len(ranges) >= 3 else None

    range_ratio = (
        prev_range_pct / atr_pct if atr_pct and atr_pct > 1e-9 else None
    )

    vol_ratio = None
    vols = [v for *_x, v in candles if v is not None and v > 0]
    if len(vols) >= 4 and _pvol is not None and _pvol > 0:
        earlier = vols[:-1]
        avg_v = sum(earlier) / len(earlier)
        if avg_v > 0:
            vol_ratio = min(10.0, _pvol / avg_v)

    # consecutive candles AGAINST the trade direction (reds before a BUY)
    opp_streak = 0
    for o, _h, _l, c, _v in reversed(candles):
        against = (c < o) if side == "BUY" else (c > o)
        if against:
            opp_streak += 1
        else:
            break
    opp_streak = min(opp_streak, 10) if candles else None

    rng = ph - pl
    rejection = against_wick = close_pos = None
    if rng > 0:
        top, bot = max(po, pc), min(po, pc)
        upper = (ph - top) / rng
        lower = (bot - pl) / rng
        rejection = lower if side == "BUY" else upper
        against_wick = upper if side == "BUY" else lower
        pos = (pc - pl) / rng
        close_pos = pos if side == "BUY" else 1.0 - pos

    trend_3 = trend_10 = price_vs_mean = None
    if len(candles) >= 3:
        o3 = candles[-3][0]
        trend_3 = (candles[-1][3] - o3) / o3 * 100.0 * sign
        mean_c = sum(c[3] for c in candles) / len(candles)
        if mean_c > 0:
            price_vs_mean = (price - mean_c) / mean_c * 100.0 * sign
    if len(candles) >= 6:
        o0 = candles[0][0]
        trend_10 = (candles[-1][3] - o0) / o0 * 100.0 * sign

    btc_trend = btc_range = None
    bc = []
    for c in (ctx or {}).get("btc_closed") or []:
        cv = _ai_candle_vals(c)
        if cv is not None:
            bc.append(cv)
    if len(bc) >= 3:
        bo = bc[-3][0]
        btc_trend = (bc[-1][3] - bo) / bo * 100.0 * sign
        btc_range = sum((h - l) / o * 100.0 for o, h, l, _c, _v in bc) / len(bc)

    share = None
    if ctx:
        share = ctx.get("green_share") if side == "BUY" else ctx.get("red_share")

    scanner_age = None
    if ctx and ctx.get("scanner_age") is not None:
        scanner_age = min(900.0, max(0.0, ctx["scanner_age"]))

    quality = str(info.get("quality") or "").upper()
    q_val = 1.0 if quality == "CONFIRMED" else (0.5 if quality == "WATCH" else 0.0)
    bias = str(info.get("bias") or "").upper()
    want = "LONG" if side == "BUY" else "SHORT"
    if not info:
        bias_match = None
    elif bias == want:
        bias_match = 1.0
    elif bias in ("LONG", "SHORT"):
        bias_match = -1.0
    else:
        bias_match = 0.0

    mark = snap.get("mark")
    mark_dev = (
        (mark - price) / price * 100.0 * sign
        if mark and mark > 0 else None
    )

    dow = now.weekday() + hour / 24.0
    dow_sin = math.sin(2.0 * math.pi * dow / 7.0)
    dow_cos = math.cos(2.0 * math.pi * dow / 7.0)

    spread_to_atr = (
        spread / atr_pct if atr_pct and atr_pct > 1e-9 else None
    )
    sl_price_pct = PRIMARY_STOP_LOSS_ROI_PCT / max(1.0, float(TRADE_LEVERAGE))
    sl_to_atr = (
        sl_price_pct / atr_pct if atr_pct and atr_pct > 1e-9 else None
    )

    setup_match = (
        1.0 if (opp_streak is not None and opp_streak >= 3 and cross_dist > 0)
        else 0.0
    )

    # ---------------- v3 features (optional) ----------------
    ex = snap.get("extra") or {}
    fr = _ai_finite(ex.get("funding"))
    funding_pct = fr * 100.0 * sign if fr is not None else None   # >0: this side pays funding
    oi = _ai_finite(ex.get("oi_value"))
    oi_log = math.log10(1.0 + max(0.0, oi)) if oi is not None else None
    p24 = _ai_finite(ex.get("pct24h"))
    pct24h = p24 * sign if p24 is not None else None
    rs_btc = (
        trend_3 - btc_trend
        if (trend_3 is not None and btc_trend is not None) else None
    )
    sym_wr = _ai_symbol_wr(snap.get("sym"))
    bot_streak = _ai_bot_streak()

    candle_progress = None
    cs = _ai_finite(snap.get("cur_start"))
    if cs:
        elapsed = (time.time() * 1000.0 - cs) / 1000.0 if cs > 1e11 else (time.time() - cs)
        candle_progress = max(0.0, min(1.5, elapsed / (float(INTERVAL) * 60.0)))

    extra = [
        atr_pct, range_ratio, vol_ratio,
        float(opp_streak) if opp_streak is not None else None,
        rejection, against_wick, close_pos,
        trend_3, trend_10, price_vs_mean,
        btc_trend, btc_range, share,
        scanner_age, q_val, bias_match,
        mark_dev, dow_sin, dow_cos,
        spread_to_atr, sl_to_atr, setup_match,
        funding_pct, oi_log, pct24h, rs_btc,
        sym_wr, bot_streak, candle_progress,
    ]
    x = [float(v) for v in base] + [_ai_finite(v) for v in extra]
    if len(x) != len(AI_FEATURE_NAMES):
        return None
    return x

def ai_build_features(symbol, side, price):
    """Entry-time feature vector (same order as AI_FEATURE_NAMES) or None."""
    snap = _ai_symbol_snapshot(symbol)
    if not snap:
        return None
    ctx = _ai_market_context()
    return _ai_features_from(snap, side, price, ctx)


# ------------------------------------------------------------
# model: logistic + boosted stumps + ridge EV
# ------------------------------------------------------------

def _ai_standardizer(rows_x, d):
    mean, std = [], []
    for j in range(d):
        vals = [r[j] for r in rows_x if r[j] is not None]
        if len(vals) >= 3:
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        else:
            m, sd = 0.0, 1.0
        mean.append(m)
        std.append(sd if sd > 1e-9 else 1.0)
    return mean, std


def _ai_zrow(x, mean, std):
    return [
        0.0 if x[j] is None
        else max(-5.0, min(5.0, (x[j] - mean[j]) / std[j]))
        for j in range(len(mean))
    ]


def _ai_fit_linear(Z, T, W, epochs, lr, lam, logistic):
    """Weighted gradient-descent GLM. logistic=True -> logit link."""
    n = len(Z)
    d = len(Z[0])
    sw = sum(W) or 1.0
    mean_t = sum(w * t for w, t in zip(W, T)) / sw
    b = _ai_logit(mean_t) if logistic else 0.0
    w = [0.0] * d
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for z, t, wi in zip(Z, T, W):
            s = b + sum(map(_mul, w, z))
            pred = _ai_sigmoid(s) if logistic else s
            e = (pred - t) * wi
            gw = [g + e * zj for g, zj in zip(gw, z)]
            gb += e
        w = [wj - lr * (g / sw + lam * wj) for wj, g in zip(w, gw)]
        b -= lr * gb / sw
        time.sleep(0)
    return w, b


def _ai_fit_gbm(Z, Y, W, rounds, lr, nbins, lam=3.0):
    """Boosted depth-1 trees on logistic loss (Newton leaf values)."""
    n = len(Z)
    d = len(Z[0])
    thr = []
    for j in range(d):
        col = sorted(z[j] for z in Z)
        ts = []
        for k in range(1, nbins):
            v = col[int(k * (n - 1) / nbins)]
            if not ts or v > ts[-1] + 1e-9:
                ts.append(v)
        thr.append([t for t in ts if t < col[-1]])
    bins = [[bisect_left(thr[j], z[j]) for j in range(d)] for z in Z]

    sw = sum(W) or 1.0
    f0 = _ai_logit(sum(w * y for w, y in zip(W, Y)) / sw)
    F = [f0] * n
    stumps = []

    for _ in range(rounds):
        G = [0.0] * n
        H = [0.0] * n
        gsum = hsum = 0.0
        for i in range(n):
            p = _ai_sigmoid(F[i])
            G[i] = (Y[i] - p) * W[i]
            H[i] = p * (1.0 - p) * W[i] + 1e-9
            gsum += G[i]
            hsum += H[i]
        min_h = max(2.0, 0.02 * hsum)
        parent = gsum * gsum / (hsum + lam)
        best = None
        for j in range(d):
            nb = len(thr[j]) + 1
            if nb < 2:
                continue
            gb = [0.0] * nb
            hb = [0.0] * nb
            for i in range(n):
                b = bins[i][j]
                gb[b] += G[i]
                hb[b] += H[i]
            GL = HL = 0.0
            for k in range(nb - 1):
                GL += gb[k]
                HL += hb[k]
                HR = hsum - HL
                if HL < min_h or HR < min_h:
                    continue
                GR = gsum - GL
                gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
                if best is None or gain > best[0]:
                    best = (gain, j, k, GL / (HL + lam), GR / (HR + lam))
        if best is None or best[0] <= 1e-9:
            break
        _, j, k, vl, vr = best
        vl *= lr
        vr *= lr
        stumps.append([j, thr[j][k], vl, vr])
        for i in range(n):
            F[i] += vl if bins[i][j] <= k else vr
        time.sleep(0)
    return f0, stumps


def _ai_gbm_logit(f0, stumps, z):
    s = f0
    for j, t, vl, vr in stumps:
        s += vl if z[int(j)] <= t else vr
    return s


def _ai_raw_logit(m, z):
    lr_logit = m["lr_b"] + sum(map(_mul, m["lr_w"], z))
    if m["gbm"]:
        g_logit = _ai_gbm_logit(m["gbm_f0"], m["gbm"], z)
        return 0.5 * lr_logit + 0.5 * g_logit, lr_logit, g_logit
    return lr_logit, lr_logit, None


def _ai_fit_platt(logits, labels):
    a, c = 1.0, 0.0
    n = len(logits)
    for _ in range(250):
        ga = gc = 0.0
        for s, y in zip(logits, labels):
            e = _ai_sigmoid(a * s + c) - y
            ga += e * s
            gc += e
        a -= 0.05 * (ga / n + 0.05 * (a - 1.0))
        c -= 0.05 * (gc / n + 0.05 * c)
    return max(0.2, min(3.0, a)), max(-3.0, min(3.0, c))


def _ai_fit_all(rows, hyper_epochs=None):
    """Fit all heads on rows -> partial model dict (no validation)."""
    d = len(AI_FEATURE_NAMES)
    mean, std = _ai_standardizer([r["x"] for r in rows], d)
    Z = [_ai_zrow(r["x"], mean, std) for r in rows]
    Y = [float(r["y"]) for r in rows]
    W = [r["w"] for r in rows]
    R = [max(-20.0, min(20.0, r["roi"])) for r in rows]

    epochs = hyper_epochs or AI_TRAIN_EPOCHS
    lr_w, lr_b = _ai_fit_linear(
        Z, Y, W, epochs, AI_LEARNING_RATE, AI_L2, True)
    gbm_f0, gbm = _ai_fit_gbm(
        Z, Y, W, AI_GBM_ROUNDS, AI_GBM_LR, AI_GBM_BINS)

    sw = sum(W) or 1.0
    mu = sum(w * r for w, r in zip(W, R)) / sw
    var = sum(w * (r - mu) ** 2 for w, r in zip(W, R)) / sw
    sd = math.sqrt(var) if var > 1e-9 else 1.0
    Tz = [(r - mu) / sd for r in R]
    ev_w, ev_b = _ai_fit_linear(
        Z, Tz, W, epochs, AI_LEARNING_RATE * 0.5, AI_L2 * 5.0, False)

    return {
        "version": AI_MODEL_VERSION, "mean": mean, "std": std,
        "lr_w": lr_w, "lr_b": lr_b, "gbm_f0": gbm_f0, "gbm": gbm,
        "ev_w": ev_w, "ev_b": ev_b, "ev_mu": mu, "ev_sd": sd,
        "plat_a": 1.0, "plat_b": 0.0,
    }


def _ai_prepare_rows(samples):
    d = len(AI_FEATURE_NAMES)
    rows = []
    for smp in samples:
        x = _ai_pad_x(smp.get("x"), d)
        if x is None:
            continue
        roi = _ai_finite(smp.get("roi"))
        y = 1 if smp.get("y") else 0
        if roi is None:
            roi = 1.0 if y else -1.0
        src = smp.get("src")
        if src not in ("trade", "setup"):
            src = "shadow"
        rows.append({
            "x": x, "y": y, "roi": roi,
            "t": _ai_finite(smp.get("t")) or 0.0, "src": src,
        })

    core = sorted((r for r in rows if r["src"] != "shadow"), key=lambda r: r["t"])
    core = core[-AI_TRAIN_MAX:]
    room = max(0, AI_TRAIN_MAX - len(core))
    k = min(room, AI_RANDOM_SHADOW_MAX_TRAIN)
    shadow = sorted((r for r in rows if r["src"] == "shadow"), key=lambda r: r["t"])
    shadow = shadow[-k:] if k > 0 else []
    rows = sorted(core + shadow, key=lambda r: r["t"])

    weight = {
        "trade": 1.0,
        "setup": AI_SETUP_SHADOW_WEIGHT,
        "shadow": AI_SHADOW_WEIGHT,
    }
    n = len(rows)
    for i, r in enumerate(rows):
        r["w"] = weight[r["src"]] * (0.5 ** ((n - 1 - i) / float(AI_RECENCY_HALFLIFE)))
    return rows


def _ai_wf_split(rows):
    """Expanding-window walk-forward folds over the newest part of the rows."""
    n = len(rows)
    n_test = int(n * AI_WF_TEST_FRACTION)
    k = AI_WF_FOLDS
    if n_test < k * 12 or (n - n_test) < AI_WF_MIN_TRAIN:
        return []
    start = n - n_test
    size = n_test // k
    folds = []
    for i in range(k):
        a = start + i * size
        b = n if i == k - 1 else a + size
        train = rows[:a]
        if len(train) > AI_WF_FOLD_MAX_ROWS:
            train = train[-AI_WF_FOLD_MAX_ROWS:]
        folds.append((train, rows[a:b]))
    return folds


def _ai_eval_oos(folds):
    """Fit on the past, score the future. Returns pooled out-of-time predictions."""
    logits, labels, rois, kinds, fold_aucs = [], [], [], [], []
    for train, test in folds:
        pos = sum(r["y"] for r in train)
        if not (0 < pos < len(train)):
            fold_aucs.append(None)
            continue
        part = _ai_fit_all(train)
        f_scores, f_labels = [], []
        for r in test:
            z = _ai_zrow(r["x"], part["mean"], part["std"])
            lg, _, _ = _ai_raw_logit(part, z)
            logits.append(lg)
            labels.append(r["y"])
            rois.append(r["roi"])
            kinds.append(r["src"])
            f_scores.append(lg)
            f_labels.append(r["y"])
        fold_aucs.append(_ai_auc(f_scores, f_labels))
    return logits, labels, rois, kinds, fold_aucs


def _ai_tune_threshold(probs, rois):
    """Cut-off that maximises total net ROI of the KEPT setups.

    Returns (threshold, keep_fraction); threshold 0.0 means 'keep everything'.
    """
    n = len(probs)
    if n < max(10, AI_MIN_TUNE_ROWS // 2):
        return None
    ordered = sorted(probs)
    best_total, best_thr, best_keep = sum(rois), 0.0, 1.0
    for qi in range(1, 14):                      # veto at most the lowest 65%
        thr = ordered[min(n - 1, int(qi * 0.05 * n))]
        kept = [r for p, r in zip(probs, rois) if p >= thr]
        if not kept or len(kept) / n < AI_TUNE_MIN_KEEP:
            continue
        total = sum(kept)
        if total > best_total + 1e-9:
            best_total, best_thr, best_keep = total, thr, len(kept) / n
    return best_thr, best_keep


def _ai_lift(probs, rois, thr):
    """(mean ROI of kept - mean ROI of all, number vetoed)."""
    if not rois:
        return None, 0
    kept = [r for p, r in zip(probs, rois) if p >= thr]
    if not kept:
        return None, 0
    return sum(kept) / len(kept) - sum(rois) / len(rois), len(rois) - len(kept)


def _ai_train_once(samples):
    rows = _ai_prepare_rows(samples)
    n = len(rows)
    if n < AI_MIN_TRAIN_SAMPLES:
        return None

    n_real = sum(1 for r in rows if r["src"] == "trade")
    wins_real = sum(r["y"] for r in rows if r["src"] == "trade")
    core = [r for r in rows if r["src"] != "shadow"]
    n_setup = len(core)
    wins_setup = sum(r["y"] for r in core)
    wins = sum(r["y"] for r in rows)
    sw = sum(r["w"] for r in rows) or 1.0
    base = sum(r["w"] * r["y"] for r in rows) / sw

    # ---- walk-forward validation (always train on the past, test on the future) ----
    val = None
    plat_a, plat_b = 1.0, 0.0
    veto_thr = None
    edge_ok = False

    if n >= AI_MIN_VALID_TOTAL:
        folds = _ai_wf_split(rows)
        if folds:
            logits, labels, rois, kinds, fold_aucs = _ai_eval_oos(folds)
            if len(logits) >= 40 and 0 < sum(labels) < len(labels):
                plat_a, plat_b = _ai_fit_platt(logits, labels)
                probs = [_ai_sigmoid(plat_a * lg + plat_b) for lg in logits]

                # judge the model on the rows it will actually be asked about
                idx = [i for i, k in enumerate(kinds) if k != "shadow"]
                if len(idx) < AI_MIN_TUNE_ROWS:
                    idx = list(range(len(probs)))
                tp = [probs[i] for i in idx]
                tl = [labels[i] for i in idx]
                tr_roi = [rois[i] for i in idx]

                auc = _ai_auc(tp, tl)
                first_test = len(rows) - sum(len(t) for _, t in folds)
                head = rows[:first_test]
                tr_base = (sum(r["y"] for r in head) / len(head)) if head else base
                brier = sum((p - y) ** 2 for p, y in zip(tp, tl)) / len(tp)
                brier_base = sum((tr_base - y) ** 2 for y in tl) / len(tl)

                half = len(tp) // 2
                lift_hold, vetoed_hold = None, 0
                if half >= 15:
                    res_a = _ai_tune_threshold(tp[:half], tr_roi[:half])
                    if res_a and res_a[0] > 0:
                        lift_hold, vetoed_hold = _ai_lift(tp[half:], tr_roi[half:], res_a[0])

                res_all = _ai_tune_threshold(tp, tr_roi)
                thr_all, keep_frac = res_all if res_all else (0.0, 1.0)
                lift_all = None
                if thr_all > 0:
                    lift_all, _ = _ai_lift(tp, tr_roi, thr_all)

                good_folds = sum(1 for a in fold_aucs if a is not None and a >= 0.5)
                edge_ok = bool(
                    auc is not None and auc >= AI_MIN_VALID_AUC
                    and brier < brier_base
                    and good_folds >= 2
                    and thr_all > 0
                    and lift_hold is not None
                    and lift_hold >= AI_TUNE_MIN_LIFT_ROI
                    and vetoed_hold >= 5
                )
                veto_thr = thr_all if thr_all > 0 else None
                val = {
                    "n": len(tp), "auc": auc, "brier": brier,
                    "brier_base": brier_base, "folds": fold_aucs,
                    "thr": veto_thr, "keep_frac": keep_frac,
                    "lift_all": lift_all, "lift_hold": lift_hold,
                    "vetoed_hold": vetoed_hold,
                    "plat_a": plat_a, "plat_b": plat_b,
                }

    model = _ai_fit_all(rows)
    model["plat_a"], model["plat_b"] = plat_a, plat_b
    model.update({
        "n": n, "wins": wins, "n_real": n_real, "wins_real": wins_real,
        "n_setup": n_setup, "wins_setup": wins_setup,
        "base_rate": base,
        "base_rate_real": (wins_real / n_real) if n_real else None,
        "base_rate_setup": (wins_setup / n_setup) if n_setup else None,
        "val": val, "edge_ok": edge_ok, "veto_thr": veto_thr,
        "trained_at": time.time(),
    })
    return model


def _ai_train_worker():
    global _ai_train_pending, _ai_train_running, ai_model
    global _ai_train_force, _ai_last_train_end, _ai_new_since_train

    while True:
        with _ai_lock:
            force = _ai_train_force
            _ai_train_force = False
        if not force:
            wait = AI_RETRAIN_MIN_SECONDS - (time.time() - _ai_last_train_end)
            if wait > 0:
                time.sleep(min(wait, AI_RETRAIN_MIN_SECONDS))

        with _ai_lock:
            _ai_train_pending = False
            _ai_new_since_train = 0
            samples = list(ai_samples)

        model = None
        try:
            model = _ai_train_once(samples)
        except Exception as exc:
            print(f"[AI] training error: {exc}")

        _ai_last_train_end = time.time()

        if model is not None:
            with _ai_lock:
                ai_model = model
            _atomic_write_json(AI_MODEL_FILE, {
                **model,
                "features": list(AI_FEATURE_NAMES),
                "updated_at": time.time(),
            })
            v = model.get("val") or {}
            auc = v.get("auc")
            print(
                f"[AI] retrained: n={model['n']} (real {model['n_real']}) "
                f"base={model['base_rate'] * 100:.1f}% "
                f"val_auc={'n/a' if auc is None else f'{auc:.3f}'} "
                f"edge_ok={model['edge_ok']}"
            )

        with _ai_lock:
            if not _ai_train_pending:
                _ai_train_running = False
                return


def ai_request_training(force=False):
    """Retrain in a background thread (never blocks trading threads)."""
    global _ai_train_pending, _ai_train_running, _ai_train_force

    with _ai_lock:
        _ai_train_pending = True
        if force:
            _ai_train_force = True
        if _ai_train_running:
            return
        _ai_train_running = True

    threading.Thread(
        target=_ai_train_worker, daemon=True, name="ai-trainer",
    ).start()


# ------------------------------------------------------------
# prediction
# ------------------------------------------------------------

def ai_predict_details(x):
    """dict(prob, ev_roi, drift, missing, lr_prob, gbm_prob) or None."""
    with _ai_lock:
        m = ai_model
    d = len(AI_FEATURE_NAMES)
    if (
        not m.get("mean") or not m.get("lr_w")
        or not isinstance(x, list) or len(x) != d
        or len(m["mean"]) != d
    ):
        return None

    xv = [_ai_finite(v) for v in x]
    z = _ai_zrow(xv, m["mean"], m["std"])
    logit, lr_logit, g_logit = _ai_raw_logit(m, z)
    prob = _ai_sigmoid(m["plat_a"] * logit + m["plat_b"])
    ev = m["ev_mu"] + m["ev_sd"] * (
        m["ev_b"] + sum(map(_mul, m["ev_w"], z)))
    seen = [abs(z[j]) for j in range(d) if xv[j] is not None]
    return {
        "prob": prob,
        "ev_roi": ev,
        "drift": (sum(seen) / len(seen)) if seen else None,
        "missing": sum(1 for v in xv if v is None),
        "lr_prob": _ai_sigmoid(lr_logit),
        "gbm_prob": _ai_sigmoid(g_logit) if g_logit is not None else None,
    }


def ai_predict(x):
    """Win probability for feature vector x, or None if no model yet."""
    det = ai_predict_details(x)
    return det["prob"] if det else None


def ai_log_prediction(symbol, side, x, prob):
    """Print the AI view of an entry (win-prob, expected ROI, drift)."""
    det = ai_predict_details(x) if x is not None else None
    if not det:
        return
    drift = det.get("drift")
    print(
        f"[AI] {symbol} {side}: win-prob {det['prob'] * 100:.1f}% | "
        f"exp ROI {det['ev_roi']:+.2f}% (after fees) | "
        f"drift {'n/a' if drift is None else f'{drift:.2f}'} | "
        f"missing {det['missing']}"
    )


def ai_should_veto(prob):
    """Return (veto, threshold, reason). Advisory-only until the edge is proven."""
    with _ai_lock:
        m = ai_model
        n = int(m.get("n_setup") or 0)
        wins = int(m.get("wins_setup") or 0)
        thr = m.get("veto_thr")
        base = m.get("base_rate_setup") or m.get("base_rate")
        edge_ok = bool(m.get("edge_ok"))

    if prob is None:
        return False, None, "no AI score"

    losses = n - wins
    if (
        n < AI_MIN_TRADES_TO_VETO
        or wins < AI_MIN_CLASS_COUNT
        or losses < AI_MIN_CLASS_COUNT
    ):
        return False, None, f"learning {n}/{AI_MIN_TRADES_TO_VETO} setup samples (advisory only)"

    if AI_REQUIRE_VALID_EDGE and not edge_ok:
        return False, None, "walk-forward edge not proven yet (advisory only)"

    if thr is None:
        if base is None:
            return False, None, "no baseline yet"
        thr = base * AI_VETO_RATIO

    if prob < thr:
        return (
            True, thr,
            f"win-prob {prob * 100:.1f}% < tuned cut-off {thr * 100:.1f}%",
        )
    return False, thr, "ok"


def ai_register_veto():
    global ai_vetoed_count
    with _ai_lock:
        ai_vetoed_count += 1


def _ai_live_auc():
    with _ai_lock:
        preds = list(_ai_live_preds)
    if len(preds) < 20:
        return None, len(preds)
    return _ai_auc([p for p, _ in preds], [y for _, y in preds]), len(preds)


def ai_status_lines():
    """Two short text lines for the dashboard (must stay 2 lines)."""
    with _ai_lock:
        m = ai_model
        n_all = int(m.get("n") or 0)
        n_real = int(m.get("n_real") or 0)
        n_setup = int(m.get("n_setup") or 0)
        wins_setup = int(m.get("wins_setup") or 0)
        thr = m.get("veto_thr")
        val = m.get("val")
        edge_ok = bool(m.get("edge_ok"))
        vetoed = ai_vetoed_count

    losses_setup = n_setup - wins_setup
    learning = (
        n_setup < AI_MIN_TRADES_TO_VETO
        or wins_setup < AI_MIN_CLASS_COUNT
        or losses_setup < AI_MIN_CLASS_COUNT
    )
    if learning:
        line1 = (
            f"AI Model  : LEARNING setups {n_setup}/{AI_MIN_TRADES_TO_VETO}"
            f" | real {n_real} | total {n_all}"
        )
    elif edge_ok and thr:
        line1 = (
            f"AI Model  : ACTIVE setups {n_setup} | real {n_real} | "
            f"veto < {thr * 100:.1f}% win-prob"
        )
    else:
        line1 = (
            f"AI Model  : ADVISORY setups {n_setup} | real {n_real} | "
            f"edge not proven yet"
        )

    if val and val.get("auc") is not None:
        q = f"WF-AUC {val['auc']:.2f}"
        lh = val.get("lift_hold")
        if lh is not None:
            q += f" lift {lh:+.2f}%"
        q += " OK" if edge_ok else " weak"
    else:
        q = "WF-AUC n/a"
    live_auc, live_n = _ai_live_auc()
    if live_auc is not None:
        q += f" live {live_auc:.2f}({live_n})"

    with _ai_lock:
        sh = dict(_ai_shadow_stats)
        pend = len(_ai_shadow_pending)
    line2 = (
        f"AI Quality: {q} | vetoed {vetoed} | "
        f"shadow {sh['closed']} done/{pend} open"
    )
    return line1, line2


# ------------------------------------------------------------
# samples: storage, learning from real trades, startup
# ------------------------------------------------------------

def _ai_src_of(smp):
    src = smp.get("src")
    return src if src in ("trade", "setup") else "shadow"


def _ai_trim_locked():
    """Drop oldest random-shadow first, then oldest setups, then oldest trades."""
    excess = len(ai_samples) - AI_MAX_SAMPLES
    if excess <= 0:
        return
    for src in ("shadow", "setup", "trade"):
        if excess <= 0:
            break
        keep = []
        for smp in ai_samples:
            if excess > 0 and _ai_src_of(smp) == src:
                excess -= 1
                _ai_sample_keys.discard(smp["key"])
            else:
                keep.append(smp)
        ai_samples[:] = keep


def _ai_add_sample(sample):
    """Thread-safe add + trim."""
    global _ai_dirty, _ai_new_since_train
    with _ai_lock:
        if sample["key"] in _ai_sample_keys:
            return False
        ai_samples.append(sample)
        _ai_sample_keys.add(sample["key"])
        _ai_trim_locked()
        _ai_dirty = True
        _ai_new_since_train += 1
    return True


def ai_flush_samples(force=False):
    global _ai_dirty, _ai_last_flush
    with _ai_lock:
        if not _ai_dirty and not force:
            return
        snapshot = list(ai_samples)
        _ai_dirty = False
        _ai_last_flush = time.time()
    _atomic_write_json(AI_SAMPLES_FILE, snapshot)


def ai_learn_from_record(record):
    """Add one closed REAL/paper trade to the training set and retrain."""
    d = len(AI_FEATURE_NAMES)
    x = _ai_pad_x(record.get("ai_features"), d)
    if record.get("mode") not in ("live", "paper") or x is None:
        return

    net = to_float(record.get("net_pnl_usdt")) or 0.0
    y = 1 if net > 0 else 0
    margin = to_float(record.get("margin"))
    if margin and margin > 0:
        roi = net / margin * 100.0
    else:
        roi = to_float(record.get("margin_roi_pct")) or (1.0 if y else -1.0)
    closed_at = to_float(record.get("closed_at")) or time.time()

    prob = to_float(record.get("ai_prob"))
    if prob is not None:
        with _ai_lock:
            _ai_live_preds.append((prob, y))
            del _ai_live_preds[:-300]

    key = f"{record.get('symbol')}|{record.get('closed_at')}"
    added = _ai_add_sample({
        "key": key, "x": x, "y": y, "roi": roi,
        "t": closed_at, "src": "trade", "sym": record.get("symbol"),
    })
    if added:
        _ai_note_symbol_outcome(record.get("symbol"), y)
        with _ai_lock:
            _ai_recent_real.append(y)
            del _ai_recent_real[:-20]
        ai_flush_samples(force=True)
        ai_request_training(force=True)


def _ai_sym_from_key(key, src):
    parts = str(key).split("|")
    if src in ("setup", "shadow") and len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else None


def ai_load_state():
    """Startup: load samples + saved model, merge logged trades, train."""
    global ai_model
    d = len(AI_FEATURE_NAMES)
    loaded = _load_json_file(AI_SAMPLES_FILE, [])
    if not isinstance(loaded, list):
        loaded = []

    merged = {}
    for item in loaded:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        x = _ai_pad_x(item.get("x"), d)
        if x is None:
            continue
        key = item["key"]
        t = _ai_finite(item.get("t"))
        if t is None:
            try:
                t = float(str(key).split("|")[-1])
            except Exception:
                t = 0.0
        src = _ai_src_of(item)
        merged[key] = {
            "key": key, "x": x,
            "y": 1 if item.get("y") else 0,
            "roi": _ai_finite(item.get("roi")),
            "t": t, "src": src,
            "sym": item.get("sym") or _ai_sym_from_key(key, src),
        }

    preds = []
    for rec in list(paper_trade_log) + list(live_trade_log):
        if not isinstance(rec, dict):
            continue
        x = _ai_pad_x(rec.get("ai_features"), d)
        if x is None:
            continue
        net = to_float(rec.get("net_pnl_usdt")) or 0.0
        y = 1 if net > 0 else 0
        margin = to_float(rec.get("margin"))
        roi = (net / margin * 100.0) if margin and margin > 0 else None
        key = f"{rec.get('symbol')}|{rec.get('closed_at')}"
        t = to_float(rec.get("closed_at")) or 0.0
        if key not in merged:
            merged[key] = {"key": key, "x": x, "y": y, "roi": roi,
                           "t": t, "src": "trade", "sym": rec.get("symbol")}
        prob = to_float(rec.get("ai_prob"))
        if prob is not None:
            preds.append((t, prob, y))

    ordered = sorted(merged.values(), key=lambda smp: smp.get("t") or 0.0)
    with _ai_lock:
        ai_samples[:] = ordered
        _ai_sample_keys.clear()
        _ai_sample_keys.update(smp["key"] for smp in ai_samples)
        _ai_trim_locked()
        snapshot = list(ai_samples)
        preds.sort(key=lambda p: p[0])
        _ai_live_preds[:] = [(p, y) for _, p, y in preds][-300:]
        _ai_sym_ema.clear()
        _ai_recent_real.clear()

    # rebuild per-symbol form and the bot's recent-trade streak
    n_trade = n_setup = 0
    for smp in snapshot:
        src = _ai_src_of(smp)
        if src == "shadow":
            continue
        _ai_note_symbol_outcome(smp.get("sym"), smp["y"])
        if src == "trade":
            n_trade += 1
            with _ai_lock:
                _ai_recent_real.append(smp["y"])
        else:
            n_setup += 1
    with _ai_lock:
        del _ai_recent_real[:-20]

    count = len(snapshot)

    # saved model (instant predictions while retraining runs)
    saved = _load_json_file(AI_MODEL_FILE, None)
    if (
        isinstance(saved, dict)
        and saved.get("version") == AI_MODEL_VERSION
        and saved.get("features") == list(AI_FEATURE_NAMES)
        and saved.get("mean") and saved.get("lr_w")
    ):
        with _ai_lock:
            ai_model = {**ai_model, **{
                k: saved[k] for k in ai_model if k in saved}}

    print(f"[AI] Loaded {count} samples ({n_trade} real trades, "
          f"{n_setup} setup-shadow, {count - n_trade - n_setup} random-shadow).")
    if count >= AI_MIN_TRAIN_SAMPLES:
        ai_request_training(force=True)


# ------------------------------------------------------------
# shadow simulator: passive, no orders, same SL/trailing/fees
# ------------------------------------------------------------

def _ai_shadow_fee_roi():
    """Round-trip fee (+GST) as ROI % of margin."""
    return (
        2.0 * PAPER_FEE_RATE * (1.0 + PAPER_FEE_GST_RATE)
        * 100.0 * float(TRADE_LEVERAGE)
    )


_ai_setup_inbox = []
_ai_setup_seen = set()
_ai_real_entry_keys = set()


def _ai_mark_real_entry(symbol, side, candle_start):
    """A setup that became a REAL trade is learned from the real outcome only."""
    with _ai_lock:
        if len(_ai_real_entry_keys) > 5000:
            _ai_real_entry_keys.clear()
        _ai_real_entry_keys.add((symbol, side, candle_start))


def ai_record_setup(symbol, side, price, candle_start):
    """
    Passive: record a crossover setup (even one the bot will skip) and
    simulate it later with the bot's own SL + trailing + fees. Never trades.
    """
    if not AI_SETUP_SHADOW_ENABLED or BOT_SHUTTING_DOWN:
        return
    key = (symbol, side, candle_start)
    with _ai_lock:
        if key in _ai_setup_seen:
            return
        if len(_ai_setup_seen) > 5000:
            _ai_setup_seen.clear()
        _ai_setup_seen.add(key)
        if len(_ai_setup_inbox) + len(_ai_shadow_pending) >= (
            AI_SETUP_MAX_PENDING + AI_SHADOW_MAX_PENDING
        ):
            return

    snap = _ai_symbol_snapshot(symbol)
    if not snap:
        return
    x = _ai_features_from(snap, side, price, _ai_market_context())
    if x is None:
        return

    entry = price
    ask, bid = snap.get("ask"), snap.get("bid")
    if side == "BUY" and ask and ask > 0:
        entry = ask
    elif side == "SELL" and bid and bid > 0:
        entry = bid

    item = {
        "symbol": symbol, "side": side, "entry": entry,
        "t_open": time.time(), "x": x, "peak": 0.0,
        "sl_roi": -float(PRIMARY_STOP_LOSS_ROI_PCT),
        "kind": "setup", "candle": candle_start,
    }
    with _ai_lock:
        _ai_setup_inbox.append(item)
        _ai_shadow_stats["opened"] += 1


def _ai_shadow_cycle():
    global _ai_shadow_pending
    if BOT_SHUTTING_DOWN:
        return

    lev = float(TRADE_LEVERAGE)
    now = time.time()

    with _ai_lock:
        _inbox = list(_ai_setup_inbox)
        del _ai_setup_inbox[:]
    if _inbox:
        _ai_shadow_pending.extend(_inbox)

    with state_lock:
        quotes = {}
        for sym, st in symbol_states.items():
            if sym not in watchlist:
                continue
            live = to_float(st.get("live_price"))
            if live and live > 0:
                quotes[sym] = (
                    live,
                    to_float(st.get("bid_price")),
                    to_float(st.get("ask_price")),
                    isinstance(st.get("previous_closed_candle"), dict),
                )

    # ---- 1) resolve pending shadow trades ----
    still = []
    fee_roi = _ai_shadow_fee_roi()
    for p in _ai_shadow_pending:
        q = quotes.get(p["symbol"])
        age = now - p["t_open"]
        if q is None:
            if age < AI_SHADOW_HORIZON_SEC:
                still.append(p)
            continue
        live, bid, ask, _ = q
        sign = 1.0 if p["side"] == "BUY" else -1.0
        roi_live = (live - p["entry"]) / p["entry"] * 100.0 * sign * lev
        p["peak"] = max(p["peak"], roi_live)
        if p["peak"] >= TRAILING_ACTIVATION_PCT:
            lock = TRAILING_INITIAL_LOCK_ROI_PCT + (
                p["peak"] - TRAILING_ACTIVATION_PCT
            ) * (1.0 - TRAILING_RETRACEMENT_PCT / 100.0)
            p["sl_roi"] = max(p["sl_roi"], lock)

        hit = roi_live <= p["sl_roi"]
        timeout = age >= AI_SHADOW_HORIZON_SEC
        if not (hit or timeout):
            still.append(p)
            continue

        exit_px = live
        if p["side"] == "BUY" and bid and bid > 0:
            exit_px = bid
        elif p["side"] == "SELL" and ask and ask > 0:
            exit_px = ask
        gross = (exit_px - p["entry"]) / p["entry"] * 100.0 * sign * lev
        net = gross - fee_roi
        kind = p.get("kind", "shadow")
        with _ai_lock:
            _dup = kind == "setup" and (
                (p["symbol"], p["side"], p.get("candle")) in _ai_real_entry_keys
            )
        if not _dup:
            if kind == "setup":
                _skey = f"setup|{p['symbol']}|{p['side']}|{p.get('candle')}"
            else:
                _skey = f"shadow|{p['symbol']}|{p['side']}|{int(p['t_open'])}"
            _added = _ai_add_sample({
                "key": _skey,
                "x": p["x"], "y": 1 if net > 0 else 0, "roi": net,
                "t": now, "src": kind, "sym": p["symbol"],
            })
            if _added and kind == "setup":
                _ai_note_symbol_outcome(p["symbol"], 1 if net > 0 else 0)
        with _ai_lock:
            _ai_shadow_stats["closed"] += 1
            if net > 0:
                _ai_shadow_stats["wins"] += 1
    _ai_shadow_pending = still

    # ---- 2) open new shadow trades (staggered, capped) ----
    if len(_ai_shadow_pending) >= AI_SHADOW_MAX_PENDING:
        return

    open_keys = {(p["symbol"], p["side"]) for p in _ai_shadow_pending}
    new_count = 0
    ctx = None
    symbols = list(quotes.keys())
    random.shuffle(symbols)
    for sym in symbols:
        if new_count >= AI_SHADOW_MAX_NEW_PER_CYCLE:
            break
        live, bid, ask, has_prev = quotes[sym]
        if not has_prev:
            continue
        for side in ("BUY", "SELL"):
            key = (sym, side)
            if key in open_keys:
                continue
            last = _ai_shadow_last.get(key)
            if last is None:
                # stagger first samples so they don't all fire at once
                _ai_shadow_last[key] = now - random.uniform(
                    0.0, AI_SHADOW_MIN_GAP_SEC)
                continue
            if now - last < AI_SHADOW_MIN_GAP_SEC:
                continue
            if len(_ai_shadow_pending) >= AI_SHADOW_MAX_PENDING:
                return
            if ctx is None:
                ctx = _ai_market_context()
            snap = _ai_symbol_snapshot(sym)
            x = _ai_features_from(snap, side, live, ctx)
            _ai_shadow_last[key] = now
            if x is None:
                continue
            entry = live
            if side == "BUY" and ask and ask > 0:
                entry = ask
            elif side == "SELL" and bid and bid > 0:
                entry = bid
            _ai_shadow_pending.append({
                "symbol": sym, "side": side, "entry": entry,
                "t_open": now, "x": x, "peak": 0.0,
                "sl_roi": -float(PRIMARY_STOP_LOSS_ROI_PCT),
                "kind": "shadow",
            })
            open_keys.add(key)
            new_count += 1
            with _ai_lock:
                _ai_shadow_stats["opened"] += 1


def ai_shadow_loop():
    while True:
        try:
            _ai_shadow_cycle()
            with _ai_lock:
                need_train = _ai_new_since_train >= AI_RETRAIN_MIN_NEW
            if need_train:
                ai_request_training()
            if time.time() - _ai_last_flush >= AI_FLUSH_SECONDS:
                ai_flush_samples()
        except Exception as exc:
            print(f"[AI] shadow loop error (ignored): {exc}")
        time.sleep(AI_SHADOW_INTERVAL)


def ai_start_shadow_thread():
    """Start the passive shadow simulator (safe to call once)."""
    global _ai_shadow_started
    if not AI_SHADOW_ENABLED or _ai_shadow_started:
        return
    _ai_shadow_started = True
    atexit.register(lambda: ai_flush_samples(force=True))
    threading.Thread(
        target=ai_shadow_loop, daemon=True, name="ai-shadow",
    ).start()
    print("[AI] Shadow learner started (passive, no orders).")


def hft_entry_gate(symbol, side):
    """Scanner confirmation/priority gate. Does not create signals."""
    info = hft_priority.get(symbol)
    if not info:
        return False, -1.0, "NO_SCANNER_DATA"

    quality = info["quality"]
    if quality not in HFT_MIN_QUALITY:
        return False, info["score"], f"QUALITY_{quality or 'UNKNOWN'}"

    expected_bias = "LONG" if side == "BUY" else "SHORT"
    if info["bias"] not in {expected_bias}:
        return False, info["score"], f"SCANNER_CONFLICT_{info['bias']}"

    if not HFT_ALLOW_WATCH_FALLBACK and quality != "CONFIRMED":
        return False, info["score"], "WATCH_NOT_ALLOWED"

    return True, info["score"], quality


def hft_scanner_loop():
    while True:
        load_hft_watchlist()
        time.sleep(HFT_REFRESH_SECONDS)


live_realized_cache = {"usdt": None, "updated": None}

_REALIZED_KEYS = {
    "realized_pnl", "realised_pnl", "realized_profit", "realised_profit",
    "realized_pnl_usdt", "realised_pnl_usdt",
}
_TX_TIME_KEYS = (
    "created_at", "timestamp", "time", "updated_at",
    "transaction_time", "executed_at",
)


def _parse_ts(value):
    try:
        if isinstance(value, (int, float)) or (
            isinstance(value, str) and value.replace(".", "", 1).isdigit()
        ):
            n = float(value)
            return n / 1000.0 if n > 1e11 else n
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None
    return None


def _collect_realized_rows(obj, out):
    """Collect (realized_pnl, timestamp_or_None) pairs from a payload."""
    if isinstance(obj, dict):
        ts = None
        for k in _TX_TIME_KEYS:
            if k in obj:
                ts = _parse_ts(obj.get(k))
                if ts is not None:
                    break
        for k, v in obj.items():
            key = str(k).lower().replace("-", "_").replace(" ", "_")
            if key in _REALIZED_KEYS:
                n = to_float(v)
                if n is not None:
                    out.append((n, ts))
            elif isinstance(v, (dict, list)):
                _collect_realized_rows(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_realized_rows(item, out)


def refresh_live_realized_cache():
    """Background-only: exchange-reported realized P&L for this session."""
    payload = get_live_transactions(limit=100)
    if payload is None:
        return False

    rows = []
    _collect_realized_rows(payload, rows)

    # Rows with a parseable timestamp older than this session are excluded.
    total = sum(
        n for n, ts in rows
        if ts is None or ts >= SESSION_STARTED_AT
    )

    with state_lock:
        live_realized_cache["usdt"] = total
        live_realized_cache["updated"] = time.time()
    return True


def balance_poll_loop():
    while True:
        if LIVE_MODE:
            refresh_account_balance()
            try:
                refresh_live_realized_cache()
            except Exception as exc:
                print(f"[LIVE P&L] realized cache refresh failed: {exc}")
        time.sleep(BALANCE_REFRESH_SECONDS)


def subscribe_kline(symbol):
    """
    Subscribes to the KLine WebSocket stream for `symbol` at
    INTERVAL minutes, and marks it as subscribed. Safe to call
    more than once for the same symbol (subscribing again is a
    harmless no-op on the server side).
    """

    try:

        sio.emit(
            "FETCH_CANDLESTICK_CS_PRO",
            {
                "event": "subscribe",
                "pair": f"{symbol}_{INTERVAL}",
            },
            namespace=NAMESPACE,
        )

    except Exception as e:

        print(f"[WebSocket] Kline subscribe error ({symbol}): {e}")


def subscribe_ticker(symbol):
    """
    Subscribes to the official Futures Ticker WebSocket stream
    (event FETCH_TICKER_INFO_CS_PRO) for `symbol`. Unlike KLines,
    the Ticker event uses the plain BASEUSDT pair (no interval
    suffix). Safe to call more than once for the same symbol.

    This runs ALONGSIDE the existing REST ticker_poll_loop() (which
    keeps polling fetch_all_pairs_ticker() as before) rather than
    replacing it - WS pushes are event-driven and typically faster,
    REST polling stays as an always-on fallback in case the socket
    drops or a particular symbol's WS push is delayed.
    """

    try:

        sio.emit(
            "FETCH_TICKER_INFO_CS_PRO",
            {
                "event": "subscribe",
                "pair": symbol,
            },
            namespace=NAMESPACE,
        )

    except Exception as e:

        print(f"[WebSocket] Ticker subscribe error ({symbol}): {e}")


# ------------------------------------------------------------
# Background candle seeding: KLines is limited to 30 req/60s per key, so
# new symbols are queued and seeded in scanner-rank order. The API guard
# paces the calls; the watchlist loop itself never blocks on them.
# ------------------------------------------------------------
_seed_queue = _queue.Queue()
_seed_pending = set()
_seed_lock = threading.Lock()
_seed_worker_started = False


def enqueue_seed(symbol):
    with _seed_lock:
        if symbol in _seed_pending:
            return
        _seed_pending.add(symbol)
    _seed_queue.put(symbol)


def _seed_worker():
    while True:
        symbol = _seed_queue.get()
        try:
            seed_symbol_history(symbol)
        except Exception as exc:
            print(f"[Seed] {symbol}: error {exc}")
        finally:
            with _seed_lock:
                _seed_pending.discard(symbol)


def start_seed_worker():
    global _seed_worker_started
    if _seed_worker_started:
        return
    _seed_worker_started = True
    threading.Thread(target=_seed_worker, daemon=True, name="seed-worker").start()


def subscribe_symbol_streams(symbol, rank=None):
    subscribe_kline(symbol)
    if rank is None or rank < WS_TICKER_TOP_N:
        subscribe_ticker(symbol)
        with state_lock:
            ws_ticker_symbols.add(symbol)


def resubscribe_all_streams():
    """After a WebSocket reconnect the server forgets every subscription."""
    with state_lock:
        symbols = sorted(subscribed_symbols, key=lambda x: symbol_rank.get(x, 10 ** 6))
        ticker_syms = set(ws_ticker_symbols)
    for symbol in symbols:
        subscribe_kline(symbol)
        if symbol in ticker_syms:
            subscribe_ticker(symbol)
        time.sleep(0.03)
        # candles missed while disconnected are refilled from REST (paced by the guard)
        enqueue_seed(symbol)
    print(f"[WebSocket] Re-subscribed {len(symbols)} symbols.")


def watchlist_refresh_loop():
    """Build the active watchlist from the HFT scanner snapshot."""
    global watchlist, last_watchlist_refresh, instrument_info
    global instrument_info_fetched_at, symbol_rank

    first_pass = True

    while True:
        try:
            load_hft_watchlist(force=True)

            if instrument_info and (time.time() - instrument_info_fetched_at) < 600:
                instrument_data = instrument_info
            else:
                instrument_data = fetch_instrument_info()
                if instrument_data:
                    instrument_info_fetched_at = time.time()

            if instrument_data:
                ranked = sorted(
                    hft_priority.keys(),
                    key=lambda sym: (
                        0 if hft_priority[sym]["quality"] == "CONFIRMED" else 1,
                        -hft_priority[sym]["score"],
                    ),
                )
                ranked = [
                    sym for sym in ranked
                    if instrument_data.get(sym, {}).get("max_leverage", 0) >= TRADE_LEVERAGE
                ]
                combined = ranked[:MAX_WATCHLIST_SYMBOLS] if MAX_WATCHLIST_SYMBOLS else ranked

                now = time.time()
                with state_lock:
                    instrument_info = instrument_data
                    new_symbols = [sym for sym in combined if sym not in subscribed_symbols]
                    for sym in new_symbols:
                        ensure_symbol_state(sym)
                        subscribed_symbols.add(sym)
                    watchlist = set(combined)
                    symbol_rank = {sym: i for i, sym in enumerate(combined)}

                    # prune symbols that left the watchlist (never one with a position)
                    pruned = []
                    for sym in list(subscribed_symbols):
                        if sym in watchlist:
                            _out_of_watchlist_since.pop(sym, None)
                            continue
                        st = symbol_states.get(sym) or {}
                        busy = (
                            st.get("in_position")
                            or st.get("position_closing")
                            or float(st.get("position_opening_until") or 0.0) > now
                        )
                        if busy:
                            _out_of_watchlist_since.pop(sym, None)
                            continue
                        since = _out_of_watchlist_since.setdefault(sym, now)
                        if now - since >= SYMBOL_PRUNE_AFTER_SEC:
                            pruned.append(sym)
                    for sym in pruned:
                        subscribed_symbols.discard(sym)
                        ws_ticker_symbols.discard(sym)
                        symbol_states.pop(sym, None)
                        _out_of_watchlist_since.pop(sym, None)

                for sym in new_symbols:
                    subscribe_symbol_streams(sym, symbol_rank.get(sym))
                    enqueue_seed(sym)
                    time.sleep(0.05)

                last_watchlist_refresh = time.time()
                if first_pass or new_symbols or pruned:
                    first_pass = False
                    print()
                    print("*" * 68)
                    print(">>> HFT WATCHLIST REFRESHED <<<")
                    print("*" * 68)
                    print(f"Scanner file       : {HFT_WATCHLIST_FILE}")
                    print(f"Scanner candidates : {len(hft_priority)}")
                    print(f"Active watchlist   : {len(combined)} (cap {MAX_WATCHLIST_SYMBOLS})")
                    print(f"New subscriptions  : {len(new_symbols)}")
                    print(f"Pruned symbols     : {len(pruned)}")
                    print("*" * 68)
            else:
                print("[Watchlist] Instrument info unavailable; keeping current watchlist.")
        except Exception as exc:
            print(f"[Watchlist] loop error (continuing): {exc}")

        time.sleep(HFT_REFRESH_SECONDS)


# ============================================================
# POSITION MANAGEMENT
# ============================================================
def calculate_raw_pnl_pct(entry, price, side):
    """Return raw price-movement PnL percentage before leverage."""
    if entry is None or entry == 0 or price is None:
        return 0.0
    if side == "BUY":
        return (price - entry) / entry * 100.0
    if side == "SELL":
        return (entry - price) / entry * 100.0
    return 0.0


def calculate_leveraged_pnl_pct(entry, price, side, leverage=None, margin=None, position_size=None):
    """
    Return position ROI/PnL percentage using actual exchange position data.

    TRADE_LEVERAGE is intentionally NOT used here.
    It is only the entry leverage setting.

    For an existing position:
      ROI % = unrealized PnL / actual position margin * 100

    When actual exchange margin is unavailable, fall back to
    position_size / actual leverage.
    """
    raw_pct = calculate_raw_pnl_pct(entry, price, side)

    if margin is not None:
        try:
            margin = float(margin)
            if margin > 0:
                if position_size is not None and float(position_size) > 0:
                    unrealized_pnl = float(position_size) * raw_pct / 100.0
                    return (unrealized_pnl / margin) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    try:
        actual_leverage = float(leverage) if leverage is not None else 0.0
        if actual_leverage > 0:
            return raw_pct * actual_leverage
    except (TypeError, ValueError):
        pass

    return 0.0



def sl_risk_allowed(entry, sl, leverage=None, symbol=None, side=None):
    """Allow entry only when leveraged SL risk is <= MAX_SL_RISK_PCT."""
    if entry is None or sl is None or entry <= 0:
        return False

    lev = float(leverage or TRADE_LEVERAGE)
    if lev <= 0:
        return False

    raw_risk_pct = abs(entry - sl) / entry * 100.0
    capital_risk_pct = raw_risk_pct * lev

    if capital_risk_pct > MAX_SL_RISK_PCT:
        label = f"{symbol} {side}" if symbol and side else (symbol or side or "trade")
        print(
            f"[ENTRY AVOIDED] {label}: SL capital risk {capital_risk_pct:.3f}% "
            f"> max {MAX_SL_RISK_PCT:.3f}% | raw price risk {raw_risk_pct:.3f}% | "
            f"leverage {lev:g}x | entry {entry:.8f} | SL {sl:.8f}"
        )
        return False

    return True

def count_open_positions():
    """Must be called under state_lock."""

    return sum(1 for s in symbol_states.values() if s["in_position"])


def total_deployed_notional():
    """Must be called under state_lock. Sum of notional (USDT) value
    across all currently open positions."""

    return sum(
        s["position_size"] or 0.0
        for s in symbol_states.values()
        if s["in_position"]
    )


def total_deployed_margin():
    """
    Must be called under state_lock.

    Sum the ACTUAL margin already deployed by every open position.

    LIVE:
      - Prefer exchange-reported position_margin.
      - If unavailable, fall back to position_size / actual_leverage.

    PAPER:
      - Use the simulated position_margin.

    This deliberately does NOT divide all positions by TRADE_LEVERAGE,
    because recovered/live positions may have exchange-reported leverage
    and margin that differ from the entry configuration.
    """

    total = 0.0

    for state in symbol_states.values():
        if not state.get("in_position"):
            continue

        margin = state.get("position_margin")

        try:
            margin = float(margin) if margin is not None else None
        except (TypeError, ValueError):
            margin = None

        if margin is not None and margin > 0:
            total += margin
            continue

        position_size = state.get("position_size")

        try:
            position_size = (
                float(position_size)
                if position_size is not None
                else None
            )
        except (TypeError, ValueError):
            position_size = None

        actual_leverage = state.get("actual_leverage")

        try:
            actual_leverage = (
                float(actual_leverage)
                if actual_leverage is not None
                else None
            )
        except (TypeError, ValueError):
            actual_leverage = None

        if (
            position_size is not None
            and position_size > 0
            and actual_leverage is not None
            and actual_leverage > 0
        ):
            total += position_size / actual_leverage

    return total


def entry_margin_allocation():
    """Return the maximum margin budget for one new entry.

    NORMAL:
      Uses ENTRY_ALLOCATION_PCT of the 50% deployment budget.

    LOW_CAPITAL_MODE:
      The 50% deployment cap is removed. One entry may use the
      full effective capital, subject to the remaining deployed
      margin and the selected leverage feasibility checks.
    """
    capital = effective_total_capital()
    if capital <= 0:
        return 0.0

    if LOW_CAPITAL_MODE:
        return capital

    max_deployment = capital * (MAX_ALLOCATION_PCT / 100.0)
    return max_deployment * (ENTRY_ALLOCATION_PCT / 100.0)



def can_afford_new_position(symbol, price, leverage=None):
    lev = float(leverage or TRADE_LEVERAGE)
    if lev <= 0 or price is None or price <= 0:
        return False

    info = instrument_info.get(symbol)
    if not info or info.get("min_qty") is None:
        return False

    min_qty = to_float(info.get("min_qty"))
    if min_qty is None or min_qty <= 0:
        return False

    prospective_notional = min_qty * price
    prospective_margin = prospective_notional / lev

    deployed_margin = total_deployed_margin()
    max_margin = max_deployable_margin()

    return (
        deployed_margin + prospective_margin
        <= max_margin + 1e-12
    )

def _recursive_dicts(value):
    """Yield every nested dict inside an API payload."""
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from _recursive_dicts(v)
    elif isinstance(value, list):
        for v in value:
            yield from _recursive_dicts(v)


def _first_number(payload, keys):
    wanted = {str(k).lower() for k in keys}
    for obj in _recursive_dicts(payload):
        for k, v in obj.items():
            if str(k).lower() in wanted:
                n = to_float(v)
                if n is not None:
                    return n
    return None


def _first_text(payload, keys):
    wanted = {str(k).lower() for k in keys}
    for obj in _recursive_dicts(payload):
        for k, v in obj.items():
            if str(k).lower() in wanted and v not in (None, ""):
                return str(v)
    return None


def _extract_order_id(payload):
    return _first_text(
        payload,
        ("order_id", "orderId", "id", "client_order_id", "clientOrderId"),
    )


def _extract_position_from_payload(payload, symbol):
    """Normalize a matching exchange position without assuming one payload shape."""
    symbol = str(symbol).upper()

    for obj in _recursive_dicts(payload):
        obj_symbol = str(
            obj.get("symbol")
            or obj.get("instrument")
            or obj.get("market")
            or ""
        ).upper()

        # STRICT POSITION MATCH:
        # A position object is valid only when it explicitly identifies
        # the requested symbol. Never accept a symbol-less wrapper,
        # aggregate object, or another symbol as this position.
        if not obj_symbol or obj_symbol != symbol:
            continue

        qty = _first_number(
            obj,
            ("position_size", "positionSize", "size", "quantity", "qty")
        )
        entry = _first_number(
            obj,
            ("avg_entry_price", "average_entry_price", "entry_price", "entryPrice")
        )

        if qty is None and entry is None:
            continue

        side = _first_text(
            obj,
            ("side", "position_side", "positionSide", "direction")
        )

        leverage = _first_number(
            obj,
            ("leverage", "actual_leverage", "position_leverage")
        )
        margin = _first_number(
            obj,
            ("position_margin", "margin", "initial_margin")
        )
        mark = _first_number(
            obj,
            ("mark_price", "markPrice", "current_price")
        )
        liquidation = _first_number(
            obj,
            ("liquidation_price", "liquidationPrice", "liq_price")
        )

        unrealised_pnl = _first_number(
            obj,
            (
                "unrealised_pnl",
                "unrealized_pnl",
                "unrealisedPnL",
                "unrealizedPnL",
                "unrealised_profit",
                "unrealized_profit",
                "unrealised_profit_loss",
                "unrealized_profit_loss",
                "pnl",
                "pnl_usdt",
            )
        )

        # Ignore obvious zero/closed position records.
        if qty is not None and abs(qty) <= 0:
            continue

        return {
            "symbol": symbol,
            "side": side,
            "qty": abs(qty) if qty is not None else None,
            "entry": entry,
            "leverage": leverage,
            "margin": margin,
            "mark": mark,
            "liquidation": liquidation,
            "unrealised_pnl": unrealised_pnl,
            "raw": obj,
        }

    return None


def _extract_order_rows(payload):
    rows = []
    for obj in _recursive_dicts(payload):
        if any(
            k in obj
            for k in (
                "order_id",
                "orderId",
                "order_type",
                "orderType",
                "trigger_price",
                "triggerPrice",
            )
        ):
            rows.append(obj)
    return rows


def _order_is_stop_protection(
    order,
    symbol,
    side,
    trigger_price,
    position_entry=None,
):
    """
    Validate an exchange-side reduce-only STOP as protective.

    Protection is intentionally direction-aware:
      LONG  -> lower trigger protects the position.
      SHORT -> higher trigger protects the position.

    An existing STOP is accepted when it is already at least as protective
    as the requested trigger. This prevents recovery from creating duplicate
    STOP orders when a stricter exchange-side STOP already exists.
    """
    if not isinstance(order, dict):
        return False

    osymbol = str(
        order.get("symbol")
        or order.get("instrument")
        or ""
    ).upper()

    if osymbol and osymbol != str(symbol).upper():
        return False

    order_type = str(
        order.get("order_type")
        or order.get("orderType")
        or ""
    ).upper()

    if "STOP" not in order_type:
        return False

    order_side = str(order.get("side") or "").upper()
    if order_side and order_side != side:
        return False

    reduce_only = order.get("reduce_only")
    if reduce_only is None:
        reduce_only = order.get("reduceOnly")

    if reduce_only is False:
        return False

    actual_trigger = _first_number(
        order,
        ("trigger_price", "triggerPrice", "stop_price", "stopPrice")
    )

    if actual_trigger is None:
        return False

    requested_trigger = float(trigger_price)
    actual_trigger = float(actual_trigger)

    tolerance = max(abs(requested_trigger) * 0.00001, 1e-10)

    # If entry is available, first require the STOP to be on the
    # protective side of the actual entry.
    if position_entry is not None:
        try:
            entry = float(position_entry)

            if side == "SELL":
                # Closing a LONG with SELL STOP: trigger must be below entry.
                if actual_trigger >= entry + tolerance:
                    return False

            elif side == "BUY":
                # Closing a SHORT with BUY STOP: trigger must be above entry.
                if actual_trigger <= entry - tolerance:
                    return False

        except (TypeError, ValueError):
            return False

    # Existing protection may be stricter than the requested trigger.
    if side == "SELL":
        # LONG protection: trigger must remain below entry.
        # A higher/equal trigger is closer to entry and therefore
        # provides at least as much protection as the requested SL.
        return actual_trigger >= requested_trigger - tolerance

    if side == "BUY":
        # SHORT protection: trigger must remain above entry.
        # A lower/equal trigger is reached earlier on an adverse rise
        # and therefore provides at least as much protection.
        return actual_trigger <= requested_trigger + tolerance

    return False


def _extract_position_payload(payload, symbol):
    return _extract_position_from_payload(payload, symbol)


def _live_position_snapshot(symbol):
    api_ok, payload = _get_live_positions_with_status(symbol)

    # API/network/signing failure:
    # position state is UNKNOWN, so never assume it is closed.
    if not api_ok:
        return None

    pos = _extract_position_payload(payload, symbol)

    # API succeeded and returned no live position for this symbol.
    # This is a confirmed external/manual close.
    if pos is None:
        return {
            "_position_closed": True,
            "symbol": str(symbol).upper(),
        }

    # Normalize exchange position side to the bot's internal convention:
    # BUY = LONG, SELL = SHORT.
    exchange_side = str(pos.get("side") or "").upper()
    if exchange_side in ("LONG", "BUY"):
        exchange_side = "BUY"
    elif exchange_side in ("SHORT", "SELL"):
        exchange_side = "SELL"
    else:
        raw_qty = pos.get("qty")
        if raw_qty is not None:
            try:
                exchange_side = "BUY" if float(raw_qty) > 0 else "SELL"
            except (TypeError, ValueError):
                exchange_side = ""

    if exchange_side in ("BUY", "SELL"):
        pos["side"] = exchange_side

    return pos


def place_exchange_stop_loss(symbol, position_side, trigger_price, quantity):
    """
    Place the primary exchange-side protective STOP_MARKET.
    BUY position -> SELL stop.
    SELL position -> BUY stop.
    """
    if not LIVE_MODE or not ORDERS_ENABLED:
        return None

    try:
        close_side = "SELL" if position_side == "BUY" else "BUY"

        headers, path = sign_request(
            "POST", "/trade/api/v2/futures/order"
        )

        body = {
            "exchange": "EXCHANGE_2",
            "symbol": str(symbol).upper(),
            "side": close_side,
            "order_type": "STOP_MARKET",
            "quantity": 0,
            "trigger_price": float(trigger_price),
            "reduce_only": True,
        }

        response = cs_post(
            BASE_URL + path,
            headers=headers,
            json=body,
            timeout=10,
        )
        response.raise_for_status()

        payload = response.json()
        order_id = _extract_order_id(payload)

        print(
            f"[PROTECTION] {symbol} STOP_MARKET "
            f"side={close_side} trigger={trigger_price:.8f} "
            f"qty=0 reduce_only=True -> {payload}"
        )

        return {
            "order_id": order_id,
            "trigger_price": float(trigger_price),
            "side": close_side,
            "raw": payload,
        }

    except requests.Timeout as exc:
        print(f"[PROTECTION AMBIGUOUS] {symbol}: STOP_MARKET timeout: {exc}")
        return {"ambiguous": True}

    except Exception as exc:
        print(f"[PROTECTION ERROR] {symbol}: {exc}")
        if "response" in locals():
            print(f"[PROTECTION HTTP] status={response.status_code}")
            print(f"[PROTECTION RESPONSE] {response.text}")
        return None


def verify_exchange_stop_loss(symbol, position_side, trigger_price):
    """
    Verify an exchange-side reduce-only STOP without making an
    additional per-symbol position REST request.

    The position entry is read from the bot's already-synchronised
    in-memory live state. This avoids adding another REST request to
    the rate-limited position endpoint during every trailing check.
    """
    payload = get_live_open_orders(symbol=symbol, limit=50)
    if payload is None:
        return None

    close_side = "SELL" if position_side == "BUY" else "BUY"

    # Use the latest exchange-confirmed entry already held in memory.
    # Do NOT call _live_position_snapshot() here: that endpoint is
    # already being rate-limited with HTTP 429 during live sync.
    position_entry = None
    try:
        with state_lock:
            state = symbol_states.get(symbol)
            if state:
                position_entry = (
                    state.get("actual_entry")
                    or state.get("position_entry")
                )
    except Exception:
        position_entry = None

    for order in _extract_order_rows(payload):
        if _order_is_stop_protection(
            order,
            symbol,
            close_side,
            trigger_price,
            position_entry=position_entry,
        ):
            verified_trigger = _first_number(
                order,
                (
                    "trigger_price",
                    "triggerPrice",
                    "stop_price",
                    "stopPrice",
                ),
            )

            return {
                "verified": True,
                "order_id": _extract_order_id(order),
                "trigger_price": verified_trigger,
                "raw": order,
            }

    return {
        "verified": False,
        "order_id": None,
    }


def _emergency_sl(entry, side, qty=None, margin=None):
    """
    Emergency SL fallback.

    EMERGENCY_SL_PCT is the maximum price-distance target.
    Actual capital risk is capped by MAX_SL_RISK_PCT whenever
    actual position quantity and margin are available.
    """
    entry = to_float(entry)
    qty = to_float(qty)
    margin = to_float(margin)

    if entry is None or entry <= 0:
        return None

    fixed_distance = entry * (EMERGENCY_SL_PCT / 100.0)

    if qty is not None and qty > 0 and margin is not None and margin > 0:
        max_loss = margin * (MAX_SL_RISK_PCT / 100.0)
        max_distance = max_loss / abs(qty)
        distance = min(fixed_distance, max_distance)
    else:
        distance = fixed_distance

    if side == "BUY":
        return entry - distance

    if side == "SELL":
        return entry + distance

    return None


def _sl_crossed(entry, side, sl, price):
    if entry is None or sl is None or price is None:
        return False
    if side == "BUY":
        return price <= sl
    return price >= sl


def _actual_sl_risk_pct(entry, sl, qty, margin):
    if not entry or not sl or not qty or not margin:
        return None

    loss = abs(entry - sl) * abs(qty)
    return (loss / margin) * 100.0


def _clear_local_position_state(symbol, recovery_status="EXTERNALLY_CLOSED"):
    """
    Centralized local position reset.

    IMPORTANT:
    This only clears LOCAL runtime state.
    It must be called only after the exchange has successfully
    confirmed that the position no longer exists.

    API failure must NEVER call this helper.
    """
    old_sl_order_id = None

    with state_lock:
        state = symbol_states.get(symbol)

        if not state:
            return None

        old_sl_order_id = state.get("sl_order_id")

        for key in (
            "in_position",
            "position_closing",
            "position_side",
            "position_entry",
            "position_sl",
            "position_size",
            "position_qty",
            "position_open_time",
            "position_margin",
            "capital_used_pct",
            "scanner_score",
            "scanner_quality",
            "scanner_bias",
            "entry_fee",
            "entry_fee_base",
            "entry_fee_gst",
            "exchange_order_id",
            "actual_entry",
            "actual_qty",
            "actual_leverage",
            "strategy_sl",
            "emergency_sl",
            "active_sl",
            "sl_order_id",
            "sl_status",
            "protection_type",
            "protection_verified_at",
            "protection_last_check",
            "mark_price",
            "liquidation_price",
            "sl_distance_pct",
            "sl_capital_risk_pct",
            "trailing_active",
            "trailing_armed",
            "trailing_stage",
            "trailing_peak",
            "trailing_sl",
            "trailing_last_order_id",
            "position_opening_until",
            "ai_features",
            "ai_prob",
        ):
            state[key] = (
                False
                if key in ("in_position", "position_closing", "trailing_active")
                else None
            )

        state["sl_status"] = "NONE"
        state["protection_type"] = "NONE"
        state["recovery_status"] = recovery_status

    return old_sl_order_id


def _finalize_external_position_close(symbol, recovery_status="EXTERNALLY_CLOSED"):
    """
    Clear a locally remembered position only after exchange-side
    reconciliation has positively confirmed zero/open-position absence.
    """
    # Do not clear a slot that open_position() reserved a moment ago and
    # whose order is still in flight / awaiting exchange confirmation.
    with state_lock:
        _s = symbol_states.get(symbol)
        if (
            _s
            and _s.get("in_position")
            and not _s.get("position_open_time")
            and float(_s.get("position_opening_until") or 0.0) > time.time()
        ):
            return False

    old_sl_order_id = _clear_local_position_state(
        symbol,
        recovery_status=recovery_status,
    )

    print(
        f"[POSITION SYNC] {symbol}: exchange confirms position CLOSED. "
        f"Local position state cleared."
    )

    # Best-effort cleanup of an old protective STOP.
    if old_sl_order_id:
        try:
            cancel_order(old_sl_order_id)
        except Exception as exc:
            print(
                f"[POSITION SYNC] {symbol}: old SL cleanup warning: {exc}"
            )

    return True


def _set_live_state_from_exchange(symbol, pos):
    """
    Update local live-position state from an exchange position.

    Accepts either:
      1) normalized position data:
         side / qty / entry / margin / leverage / mark / liquidation
      2) raw CoinSwitch futures positions row:
         position_side / position_size / avg_entry_price /
         position_margin / leverage / mark_price /
         liquidation_price / unrealised_pnl

    Exchange position data is the source of truth.
    """
    # Preserve verified exchange protection across every position sync.
    #
    # The exchange-position endpoint does not carry our local protection
    # metadata.  Therefore a successful protection verification must never
    # be lost merely because the position was refreshed from the exchange.
    #
    # Protection is preserved only for the same symbol and same position
    # identity (side + entry).  A genuinely new position gets fresh state.
    _preserved_protection = {}
    try:
        with state_lock:
            _old_state = symbol_states.get(symbol) or {}

            _old_entry = (
                _old_state.get("actual_entry")
                if _old_state.get("actual_entry") is not None
                else _old_state.get("position_entry")
            )
            _old_side = str(
                _old_state.get("position_side") or ""
            ).upper()

            _new_entry = (
                pos.get("avg_entry_price")
                if pos.get("avg_entry_price") is not None
                else (
                    pos.get("entry_price")
                    if pos.get("entry_price") is not None
                    else pos.get("entry")
                )
            )
            _new_side = str(
                pos.get("position_side")
                or pos.get("side")
                or ""
            ).upper()

            _same_position = False

            if _old_entry is not None and _new_entry is not None:
                try:
                    _same_position = (
                        _old_side == _new_side
                        and abs(float(_old_entry) - float(_new_entry))
                        <= max(abs(float(_new_entry)) * 1e-8, 1e-12)
                    )
                except (TypeError, ValueError):
                    _same_position = False

            if (
                _same_position
                and _old_state.get("sl_status") == "VERIFIED"
                and _old_state.get("recovery_status") == "PROTECTED"
                and _old_state.get("sl_order_id")
            ):
                for _key in (
                    "strategy_sl",
                    "emergency_sl",
                    "position_sl",
                    "active_sl",
                    "sl_order_id",
                    "sl_status",
                    "protection_type",
                    "protection_verified_at",
                    "protection_last_check",
                    "recovery_status",
                    "sl_distance_pct",
                    "sl_capital_risk_pct",
                ):
                    if _old_state.get(_key) is not None:
                        _preserved_protection[_key] = _old_state.get(_key)

    except Exception:
        _preserved_protection = {}

    with state_lock:
        ensure_symbol_state(symbol)
        state = symbol_states[symbol]

        # --------------------------------------------------------
        # Normalize raw CoinSwitch fields when present.
        # --------------------------------------------------------
        exchange_side = str(
            pos.get("side")
            or pos.get("position_side")
            or pos.get("positionSide")
            or ""
        ).upper()

        if exchange_side in ("LONG", "BUY"):
            exchange_side = "BUY"
        elif exchange_side in ("SHORT", "SELL"):
            exchange_side = "SELL"

        raw_qty = (
            pos.get("qty")
            if pos.get("qty") is not None
            else pos.get("position_size")
        )

        raw_entry = (
            pos.get("entry")
            if pos.get("entry") is not None
            else pos.get("avg_entry_price")
        )

        raw_margin = (
            pos.get("margin")
            if pos.get("margin") is not None
            else pos.get("position_margin")
        )

        raw_leverage = (
            pos.get("leverage")
            if pos.get("leverage") is not None
            else pos.get("actual_leverage")
        )

        raw_mark = (
            pos.get("mark")
            if pos.get("mark") is not None
            else (
                pos.get("mark_price")
                if pos.get("mark_price") is not None
                else pos.get("last_price")
            )
        )

        raw_liquidation = (
            pos.get("liquidation")
            if pos.get("liquidation") is not None
            else pos.get("liquidation_price")
        )

        raw_unrealised = (
            pos.get("unrealised_pnl")
            if pos.get("unrealised_pnl") is not None
            else pos.get("unrealized_pnl")
        )

        # --------------------------------------------------------
        # Normalize side from signed quantity if needed.
        # --------------------------------------------------------
        if exchange_side not in ("BUY", "SELL") and raw_qty is not None:
            try:
                exchange_side = (
                    "BUY" if float(raw_qty) > 0 else "SELL"
                )
            except (TypeError, ValueError):
                exchange_side = ""

        # --------------------------------------------------------
        # Store exchange position as local source of truth.
        # --------------------------------------------------------
        state["in_position"] = True
        state["position_closing"] = False
        state["position_side"] = exchange_side

        if raw_entry is not None:
            try:
                entry = float(raw_entry)
                state["position_entry"] = entry
                state["actual_entry"] = entry
            except (TypeError, ValueError):
                pass

        if raw_qty is not None:
            try:
                qty = abs(float(raw_qty))
                state["position_qty"] = qty
                state["actual_qty"] = qty
            except (TypeError, ValueError):
                pass

        if raw_margin is not None:
            try:
                margin = float(raw_margin)
                if margin > 0:
                    state["position_margin"] = margin
            except (TypeError, ValueError):
                pass

        if raw_leverage is not None:
            try:
                leverage = float(raw_leverage)
                if leverage > 0:
                    state["actual_leverage"] = leverage
            except (TypeError, ValueError):
                pass

        if raw_mark is not None:
            try:
                mark = float(raw_mark)
                if mark > 0:
                    state["mark_price"] = mark
            except (TypeError, ValueError):
                pass

        if raw_liquidation is not None:
            try:
                liquidation = float(raw_liquidation)
                if liquidation > 0:
                    state["liquidation_price"] = liquidation
            except (TypeError, ValueError):
                pass

        # CoinSwitch provides exchange-calculated unrealised PnL.
        # Preserve it separately so dashboard/live PnL can use the
        # exchange value without replacing the existing strategy math.
        if raw_unrealised is not None:
            try:
                state["exchange_unrealised_pnl"] = float(raw_unrealised)
            except (TypeError, ValueError):
                pass

        # Position notional/value used by existing capacity/dashboard
        # logic. Prefer actual entry * actual quantity because these
        # are the normalized position fields already used elsewhere.
        if state.get("position_entry") and state.get("position_qty"):
            state["position_size"] = (
                state["position_entry"] * state["position_qty"]
            )
    # Restore verified protection for the same unchanged live position.
    if _preserved_protection:
        try:
            with state_lock:
                _state = symbol_states.get(symbol)
                if _state is not None:
                    _state.update(_preserved_protection)
        except Exception:
            pass


def _protect_live_position(symbol, allow_emergency=True):
    """
    Reconcile one live exchange position and guarantee that an exchange-side
    protective stop is either verified or the position is actively recovered.
    """
    global RECOVERY_LOCK

    pos = _live_position_snapshot(symbol)

    # ------------------------------------------------------------
    # EXTERNAL / MANUAL CLOSE RECONCILIATION
    # Exchange API succeeded and confirmed that this symbol has
    # no open position. Clear the local runtime position state.
    # Do NOT treat this as an API failure.
    # ------------------------------------------------------------
    if isinstance(pos, dict) and pos.get("_position_closed"):
        _finalize_external_position_close(
            symbol,
            recovery_status="EXTERNALLY_CLOSED",
        )

        RECOVERY_LOCK = False
        return True

    if pos is None:
        with state_lock:
            state = symbol_states.get(symbol)
            if state and state.get("in_position"):
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
        RECOVERY_LOCK = True
        return False

    if pos.get("qty") is None or abs(float(pos.get("qty") or 0.0)) <= 0:
        return False

    _set_live_state_from_exchange(symbol, pos)

    with state_lock:
        state = symbol_states[symbol]
        side = state.get("position_side")
        entry = state.get("actual_entry") or state.get("position_entry")
        strategy_sl = state.get("strategy_sl") or state.get("position_sl")
        qty = state.get("actual_qty") or state.get("position_qty")
        margin = state.get("position_margin")
        mark = state.get("mark_price") or state.get("live_price")
        leverage = state.get("actual_leverage")

    if not side or not entry or not qty:
        with state_lock:
            state["recovery_status"] = "RECOVERING"
            state["sl_status"] = "UNKNOWN"
        RECOVERY_LOCK = True
        return False

    # Primary strategy SL remains primary whenever it is valid for the side.
    primary_valid = (
        strategy_sl is not None
        and (
            (side == "BUY" and strategy_sl < entry)
            or (side == "SELL" and strategy_sl > entry)
        )
    )

    target_sl = strategy_sl if primary_valid else None
    protection_type = "PRIMARY"

    # IMPORTANT:
    # During recovery, first accept an already-existing exchange-side
    # protective STOP if it is valid and protective for the recovered
    # position. Only fall back to the emergency SL when no valid
    # exchange protection exists.
    if target_sl is not None:
        verification = verify_exchange_stop_loss(
            symbol,
            side,
            target_sl,
        )
    else:
        _probe_sl = _emergency_sl(entry, side, qty, margin)
        if _probe_sl is None:
            verification = {"verified": False, "order_id": None}
        else:
            verification = verify_exchange_stop_loss(
                symbol,
                side,
                _probe_sl,
            )

    if verification is None:
        with state_lock:
            state["recovery_status"] = "RECOVERING"
            state["sl_status"] = "UNKNOWN"
        RECOVERY_LOCK = True
        return False

    if verification.get("verified"):
        verified_trigger = verification.get("trigger_price")
        try:
            verified_trigger = float(verified_trigger)
        except (TypeError, ValueError):
            verified_trigger = None

        if verified_trigger is not None:
            target_sl = verified_trigger

        protection_type = "PRIMARY" if primary_valid else "EXCHANGE_EXISTING"

    else:
        if primary_valid:
            # If the primary 3.80% ROI stop is valid but not yet present on
            # the exchange, place that exact stop; do not substitute a wider
            # emergency-price stop during ordinary entry protection.
            target_sl = strategy_sl
            protection_type = "PRIMARY"
        else:
            if not allow_emergency:
                with state_lock:
                    state["recovery_status"] = "UNPROTECTED"
                    state["sl_status"] = "INVALID"
                RECOVERY_LOCK = True
                return False

            target_sl = _emergency_sl(entry, side, qty, margin)
            protection_type = "EMERGENCY"
            if target_sl is None:
                with state_lock:
                    state["recovery_status"] = "UNPROTECTED"
                    state["sl_status"] = "INVALID"
                RECOVERY_LOCK = True
                return False
            risk_pct = _actual_sl_risk_pct(entry, target_sl, qty, margin)
            if risk_pct is not None and risk_pct > (MAX_SL_RISK_PCT + 1e-9):
                print(
                    f"[PROTECTION BLOCKED] {symbol}: {protection_type} SL risk "
                    f"{risk_pct:.3f}% exceeds {MAX_SL_RISK_PCT:.3f}%."
                )
                with state_lock:
                    state["recovery_status"] = "EMERGENCY_RISK_EXCEEDED"
                    state["sl_status"] = "INVALID"
                RECOVERY_LOCK = True
                return False

        # If the price has already crossed the required protection level,
        # close immediately instead of placing a stop that is already breached.
        if _sl_crossed(entry, side, target_sl, mark):
            print(
                f"[{protection_type} EXIT] {symbol}: current price {mark} already "
                f"crossed required SL {target_sl:.8f}. Closing reduce-only."
            )
            with state_lock:
                state["position_sl"] = target_sl
                state["strategy_sl"] = strategy_sl
                state["emergency_sl"] = target_sl if protection_type == "EMERGENCY" else None
                state["active_sl"] = target_sl
                state["protection_type"] = f"{protection_type}_BREACHED"
                state["sl_status"] = "BREACHED"
                state["recovery_status"] = "CLOSING_RECOVERED"
            close_position(symbol, mark, f"{protection_type} SL BREACHED DURING RECOVERY")
            RECOVERY_LOCK = True
            return False

        print(
            f"[PROTECTION] {symbol}: no verified exchange STOP found. "
            f"Placing {protection_type.lower()} protection at {target_sl:.8f}..."
        )
        result = place_exchange_stop_loss(symbol, side, target_sl, qty)
        if result is None or result.get("ambiguous"):
            with state_lock:
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return False

        verification = verify_exchange_stop_loss(symbol, side, target_sl)
        if verification is None or not verification.get("verified"):
            with state_lock:
                state["recovery_status"] = "UNVERIFIED"
                state["sl_status"] = "UNVERIFIED"
            RECOVERY_LOCK = True
            return False

        verified_trigger = verification.get("trigger_price")
        try:
            verified_trigger = float(verified_trigger)
        except (TypeError, ValueError):
            verified_trigger = target_sl
        target_sl = verified_trigger

    # Calculate risk from the actual protection currently being used.
    risk_pct = _actual_sl_risk_pct(entry, target_sl, qty, margin)

    with state_lock:
        state = symbol_states[symbol]
        state["strategy_sl"] = strategy_sl
        state["emergency_sl"] = _emergency_sl(entry, side, qty, margin)
        state["position_sl"] = target_sl
        state["active_sl"] = target_sl
        state["sl_order_id"] = verification.get("order_id")
        state["sl_status"] = "VERIFIED"
        state["protection_type"] = protection_type
        state["protection_verified_at"] = time.time()
        state["protection_last_check"] = time.time()
        state["recovery_status"] = "PROTECTED"
        state["actual_entry"] = entry
        state["actual_qty"] = qty
        state["actual_leverage"] = leverage
        state["sl_distance_pct"] = (
            abs(entry - target_sl) / entry * 100.0
            if entry else None
        )
        state["sl_capital_risk_pct"] = risk_pct

    RECOVERY_LOCK = False
    print(
        f"[PROTECTED] {symbol} {side}: "
        f"{protection_type} SL={target_sl:.8f} "
        f"order_id={verification.get('order_id')}"
    )
    return True


def recover_live_positions(allow_orders_off=False):
    """
    Startup reconciliation. Exchange is the source of truth.

    Normal startup recovery requires LIVE_MODE + ORDERS_ENABLED because
    recovered positions must be protected before live execution is unlocked.

    When allow_orders_off=True, R hotkey can perform a READ-ONLY live
    exchange discovery test with LIVE_MODE ON and ORDERS_ENABLED OFF.
    In that mode no protection/order action is attempted.
    """
    global RECOVERY_LOCK

    if not LIVE_MODE:
        RECOVERY_LOCK = False
        return True

    # LIVE_MODE is the only startup gate for exchange reconciliation.
    # When LIVE_MODE is enabled, always recover/reconcile the exchange
    # position state before allowing the normal strategy flow to proceed.

    readonly_recovery = bool(allow_orders_off and not ORDERS_ENABLED)

    if readonly_recovery:
        print("[RECOVERY] READ-ONLY LIVE exchange discovery mode.")
    else:
        RECOVERY_LOCK = True
        print("[RECOVERY] Starting LIVE exchange reconciliation...")

    path = "/trade/api/v2/futures/positions"

    try:
        headers, signed_path = sign_request(
            "GET",
            path,
            params={"exchange": "EXCHANGE_2"},
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[RECOVERY POSITIONS] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            print("[RECOVERY] Invalid positions payload.")
            RECOVERY_LOCK = True
            return False

        rows = payload.get("data")

        # CoinSwitch can return HTTP 200 with a message instead of
        # an empty data list when there are no open positions.
        # This is a successful, confirmed empty exchange snapshot.
        if not isinstance(rows, list):
            message = str(payload.get("message") or "").strip().lower()

            if message == "there are no open positions":
                rows = []
                print(
                    "[RECOVERY] Exchange explicitly reports no open "
                    "positions. Treating snapshot as EMPTY."
                )
            else:
                print("[RECOVERY] Positions payload missing data list.")
                RECOVERY_LOCK = True
                return False

        exchange_symbols = set()
        recovered_symbols = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            symbol = str(row.get("symbol") or "").upper().strip()
            status = str(row.get("status") or "").upper()

            try:
                qty = abs(float(row.get("position_size") or 0.0))
            except Exception:
                qty = 0.0

            if not symbol or qty <= 0 or status not in ("OPEN", "ACTIVE", ""):
                continue

            exchange_symbols.add(symbol)
            recovered_symbols.append(symbol)

            print(
                f"[RECOVERY POSITION] {symbol}: "
                f"side={row.get('position_side')} "
                f"qty={row.get('position_size')} "
                f"entry={row.get('avg_entry_price')} "
                f"leverage={row.get('leverage')}"
            )

            _set_live_state_from_exchange(symbol, row)

            if readonly_recovery:
                with state_lock:
                    state = symbol_states[symbol]
                    state["recovery_status"] = "DISCOVERED_READONLY"
                    state["sl_status"] = "UNKNOWN"

                print(
                    f"[RECOVERY READONLY] {symbol}: "
                    "exchange position restored locally; "
                    "protection check skipped because ORDERS_ENABLED=OFF."
                )
                continue

            with state_lock:
                state = symbol_states[symbol]

                # Preserve any already-known strategy SL.
                # On a restart this may be unavailable, so protection
                # verification must rely on the exchange-side STOP.
                if not state.get("strategy_sl"):
                    state["strategy_sl"] = state.get("position_sl")

                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"

            try:
                protected = _protect_live_position(symbol)
            except Exception as exc:
                print(f"[RECOVERY PROTECTION ERROR] {symbol}: {exc}")
                protected = False

            if protected:
                print(f"[RECOVERY PROTECTED] {symbol}")
            else:
                print(f"[RECOVERY UNPROTECTED] {symbol}")

        # IMPORTANT:
        # The positions endpoint succeeded and returned the complete
        # exchange position snapshot. Therefore a locally-open position
        # absent from exchange_symbols is confirmed closed externally.
        #
        # Never leave such a position as RECOVERING, because dashboard
        # and position-capacity logic use local in_position=True and would
        # otherwise create a permanent ghost position.
        with state_lock:
            local_live_symbols = {
                str(symbol).upper()
                for symbol, state in symbol_states.items()
                if state.get("in_position")
            }

        missing_locally = local_live_symbols - exchange_symbols

        for symbol in sorted(missing_locally):
            print(
                f"[RECOVERY SYNC] {symbol}: local position not found in "
                f"successful exchange snapshot. Clearing local position."
            )

            _finalize_external_position_close(
                symbol,
                recovery_status="EXTERNALLY_CLOSED",
            )

        # Final safety gate.
        # At this point every locally remembered live position has either:
        #   1) been found on the successful exchange snapshot, or
        #   2) been confirmed absent and locally cleared.
        with state_lock:
            active = [
                symbol
                for symbol, state in symbol_states.items()
                if state.get("in_position")
            ]

            all_protected = all(
                symbol in exchange_symbols
                and symbol_states[symbol].get("sl_status") == "VERIFIED"
                and symbol_states[symbol].get("recovery_status") == "PROTECTED"
                for symbol in active
            )

        RECOVERY_LOCK = not all_protected

        print(
            f"[RECOVERY] Exchange positions discovered: "
            f"{', '.join(recovered_symbols) if recovered_symbols else 'NONE'}"
        )
        print(
            f"[RECOVERY] FINAL STATUS: "
            f"{'PROTECTED / UNLOCKED' if not RECOVERY_LOCK else 'LOCKED / RECOVERING'}"
        )

        return not RECOVERY_LOCK

    except Exception as exc:
        print(f"[RECOVERY ERROR] {exc}")
        RECOVERY_LOCK = True
        return False


def start_live_position_sync_thread():
    """Start fast live position synchronization exactly once."""
    global LIVE_POSITION_SYNC_THREAD

    try:
        if LIVE_POSITION_SYNC_THREAD is not None:
            if LIVE_POSITION_SYNC_THREAD.is_alive():
                return

        LIVE_POSITION_SYNC_THREAD = threading.Thread(
            target=live_position_sync_loop,
            daemon=True,
            name="live-position-sync",
        )
        LIVE_POSITION_SYNC_THREAD.start()
        print("[LIVE SYNC] Fast live position sync started.")

    except Exception as exc:
        print(f"[LIVE SYNC] Failed to start sync thread: {exc}")


def start_protection_watchdog_thread():
    """Start the protection watchdog exactly once (idempotent)."""
    global PROTECTION_WATCHDOG_THREAD

    try:
        if PROTECTION_WATCHDOG_THREAD is not None:
            if PROTECTION_WATCHDOG_THREAD.is_alive():
                return

        PROTECTION_WATCHDOG_THREAD = threading.Thread(
            target=live_protection_watchdog,
            daemon=True,
            name="live-protection-watchdog",
        )
        PROTECTION_WATCHDOG_THREAD.start()
        print("[WATCHDOG] Protection watchdog started.")

    except Exception as exc:
        print(f"[WATCHDOG] Failed to start watchdog thread: {exc}")


def live_position_sync_loop():
    """
    Read-only exchange position synchronization with rate-limit backoff.

    Purpose:
      - Keep local in_position state aligned with exchange.
      - Detect external/manual/SL closures.
      - Do NOT perform protection placement here.
      - Do NOT unlock RECOVERY_LOCK here.

    The exchange positions endpoint returns the complete open-position set,
    so one bulk request is used per sync cycle.  On API failure (including
    HTTP 429), local state is preserved and the next request is delayed
    with exponential backoff to avoid hammering the endpoint.
    """
    global RECOVERY_LOCK

    sync_failure_count = 0

    while True:
        sleep_seconds = LIVE_POSITION_SYNC_INTERVAL

        try:
            if LIVE_MODE:

                with state_lock:
                    active_symbols = [
                        str(symbol).upper()
                        for symbol, state in symbol_states.items()
                        if state.get("in_position")
                    ]

                # The exchange positions endpoint returns the complete
                # open-position set. Fetch it ONCE per sync cycle.
                api_ok, payload = _get_all_live_positions_with_status()

                # API failure (including HTTP 429):
                # NEVER clear local state. Back off before retrying.
                if not api_ok:
                    sync_failure_count += 1
                    sleep_seconds = min(
                        LIVE_POSITION_SYNC_INTERVAL
                        * (2 ** min(sync_failure_count, 4)),
                        60.0,
                    )
                    print(
                        f"[FAST POSITION SYNC] API failure; "
                        f"preserving local state. "
                        f"failure_count={sync_failure_count} "
                        f"next_retry_in={sleep_seconds:.1f}s"
                    )
                    time.sleep(sleep_seconds)
                    continue

                # Successful bulk response: reset rate-limit backoff.
                sync_failure_count = 0

                for symbol in active_symbols:
                    try:
                        # Extract only the explicitly matching symbol from
                        # the already-fetched bulk payload.
                        pos = _extract_position_payload(payload, symbol)

                        # Bulk API succeeded and explicitly confirmed that
                        # this symbol has no open position.
                        if pos is None:
                            _finalize_external_position_close(
                                symbol,
                                recovery_status="EXTERNALLY_CLOSED",
                            )
                            print(
                                f"[FAST POSITION SYNC] {symbol}: "
                                f"exchange position closed; local state cleared."
                            )
                            continue

                        # Exchange position still exists.
                        qty = abs(float(pos.get("qty") or 0.0))
                        if qty <= 0:
                            continue

                        _set_live_state_from_exchange(symbol, pos)

                        if not ORDERS_ENABLED:
                            with state_lock:
                                state = symbol_states.get(symbol)
                                if state and state.get("in_position"):
                                    state["recovery_status"] = "DISCOVERED_READONLY"
                                    state["sl_status"] = "UNKNOWN"

                    except Exception as exc:
                        print(
                            f"[FAST POSITION SYNC ERROR] "
                            f"{symbol}: {exc}"
                        )

            elif not LIVE_MODE:
                RECOVERY_LOCK = False

        except Exception as exc:
            sync_failure_count += 1
            sleep_seconds = min(
                LIVE_POSITION_SYNC_INTERVAL
                * (2 ** min(sync_failure_count, 4)),
                60.0,
            )
            print(
                f"[FAST POSITION SYNC LOOP ERROR] {exc}; "
                f"next_retry_in={sleep_seconds:.1f}s"
            )

        time.sleep(sleep_seconds)


def live_protection_watchdog():
    """Continuously reconcile every locally open live position."""
    global RECOVERY_LOCK

    while True:
        try:
            if LIVE_MODE and ORDERS_ENABLED:
                active = []

                with state_lock:
                    for symbol, state in symbol_states.items():
                        if state.get("in_position"):
                            active.append(symbol)

                if active:
                    all_protected = True

                    for symbol in active:
                        with state_lock:
                            _ws = symbol_states.get(symbol) or {}
                            _replacing = bool(_ws.get("trailing_replacing"))
                            _fresh = (
                                _ws.get("sl_status") == "VERIFIED"
                                and _ws.get("recovery_status") == "PROTECTED"
                                and (
                                    time.time()
                                    - float(_ws.get("protection_last_check") or 0.0)
                                ) < 45.0
                            )

                        # A trailing STOP swap is in flight, or protection was
                        # verified very recently: do not hammer the REST API
                        # or place a competing STOP.
                        if _replacing or _fresh:
                            continue

                        try:
                            protected = _protect_live_position(symbol)
                            if not protected:
                                all_protected = False
                        except Exception as exc:
                            print(f"[WATCHDOG ERROR] {symbol}: {exc}")
                            all_protected = False

                    # Unlock only when every active live position has verified
                    # protection. A single failed symbol keeps the bot locked.
                    RECOVERY_LOCK = not all_protected
                else:
                    # No locally remembered positions remain.
                    # Do not keep a stale recovery/protection lock active.
                    RECOVERY_LOCK = False
                    print("[WATCHDOG] No active local positions. RECOVERY_LOCK=FALSE.")

        except Exception as exc:
            print(f"[WATCHDOG LOOP ERROR] {exc}")
            # Keep the safety lock on genuine watchdog errors.
            RECOVERY_LOCK = True

        time.sleep(POSITION_RECOVERY_INTERVAL)


def low_capital_leverage_candidates(symbol):
    with state_lock:
        info = dict(instrument_info.get(symbol) or {})

    max_lev = to_float(info.get("max_leverage")) or 0.0

    return [
        float(lev)
        for lev in LOW_CAPITAL_LEVERAGES
        if float(lev) <= max_lev + 1e-9
    ]


def normal_entry_leverage(symbol):
    """
    NORMAL MODE leverage selection.

    M ON  -> use this symbol's exchange max_leverage.
    M OFF -> use configured TRADE_LEVERAGE (10x).

    C/LOW CAPITAL MODE uses its own separate logic.
    """
    if not MAX_LEVERAGE_MODE:
        return float(TRADE_LEVERAGE)

    with state_lock:
        info = dict(instrument_info.get(symbol) or {})

    max_lev = to_float(info.get("max_leverage"))
    if max_lev is None or max_lev <= 0:
        return float(TRADE_LEVERAGE)

    return max_lev


def select_low_capital_entry(symbol, entry, sl, side):
    """
    LOW CAPITAL MODE:
      - quantity is ALWAYS exchange min_qty
      - try 10x, then 20x, then 40x
      - never exceed exchange max leverage
      - never exceed our 40x hard limit
      - selected leverage must pass SL-risk
      - selected leverage must make min_qty affordable
    """
    with state_lock:
        info = dict(instrument_info.get(symbol) or {})

    min_qty = to_float(info.get("min_qty"))
    if min_qty is None or min_qty <= 0:
        return None, None, None

    candidates = low_capital_leverage_candidates(symbol)
    if not candidates:
        return None, None, None

    with state_lock:
        deployed_margin = total_deployed_margin()

    max_margin = max_deployable_margin()
    remaining_margin = max(
        0.0,
        max_margin - deployed_margin
    )

    available = effective_available_capital()
    if available is not None and available >= 0:
        remaining_margin = min(
            remaining_margin,
            float(available)
        )

    notional = min_qty * entry

    for lev in candidates:
        lev_sl = _roi_price_for_target(
            entry, side, -PRIMARY_STOP_LOSS_ROI_PCT, leverage=lev,
            margin=notional / lev, position_size=notional,
        )
        if lev_sl is None or not sl_risk_allowed(
            entry,
            lev_sl,
            leverage=lev,
            symbol=symbol,
            side=side,
        ):
            continue

        required_margin = notional / lev

        if required_margin <= remaining_margin + 1e-12:
            return lev, min_qty, required_margin

        print(
            f"[LOW CAPITAL] {symbol}: {lev:g}x cannot afford "
            f"min_qty={min_qty:.12g} | required margin="
            f"{required_margin:.8f} | available capacity="
            f"{remaining_margin:.8f}"
        )

    return None, None, None

def risk_entry_gate(symbol, side):
    """(ok, reason). Blocks NEW entries only. Exits/stops are never affected."""
    if not RISK_GATE_ENABLED:
        return True, "off"
    try:
        if LIVE_MODE:
            history = live_trade_log
        elif PAPER_MODE:
            history = paper_trade_log
        else:
            return True, "n/a"

        rows = [
            r for r in list(history)
            if isinstance(r, dict) and to_float(r.get("closed_at")) is not None
        ]
        rows.sort(key=lambda r: float(r["closed_at"]))
        now = time.time()
        capital = effective_total_capital()

        if capital and capital > 0:
            net24 = sum(
                float(r.get("net_pnl_usdt") or 0.0)
                for r in rows if float(r["closed_at"]) >= now - 86400.0
            )
            if net24 <= -capital * RISK_DAILY_LOSS_PCT / 100.0:
                return False, (
                    f"24h net {net24:+.2f} USDT <= -{RISK_DAILY_LOSS_PCT:.1f}% of capital"
                )

        streak = 0
        for r in reversed(rows):
            if float(r.get("net_pnl_usdt") or 0.0) > 0:
                break
            streak += 1
        if rows and streak >= RISK_MAX_CONSEC_LOSSES:
            since_min = (now - float(rows[-1]["closed_at"])) / 60.0
            if since_min < RISK_PAUSE_MINUTES:
                return False, (
                    f"{streak} losses in a row; paused "
                    f"{RISK_PAUSE_MINUTES - since_min:.0f} more min"
                )

        with state_lock:
            same_side = sum(
                1 for st in symbol_states.values()
                if st.get("in_position") and st.get("position_side") == side
            )
        if same_side >= RISK_MAX_SAME_SIDE:
            return False, f"{same_side} {side} positions already open (max {RISK_MAX_SAME_SIDE})"

        return True, "ok"
    except Exception as exc:
        print(f"[RISK] gate error (ignored): {exc}")
        return True, "error"


def risk_status_line():
    if not RISK_GATE_ENABLED:
        return "Risk Gate : OFF"
    ok, reason = risk_entry_gate("", "BUY")
    return f"Risk Gate : {'OK' if ok else 'BLOCKING NEW ENTRIES - ' + reason}"


def symbol_funding_blocked(symbol):
    """Per-symbol funding blackout from the ticker's own next_funding_timestamp."""
    ex = ticker_extra.get(symbol) or {}
    nxt = ex.get("next_funding_ms")
    if not nxt:
        return False, None
    try:
        secs = (float(nxt) / 1000.0) - time.time()
    except (TypeError, ValueError):
        return False, None
    if 0.0 <= secs <= FUNDING_BLACKOUT_MINUTES * 60.0:
        return True, secs / 60.0
    return False, None


def open_position(symbol, side, entry, sl, candle_start):
    global RECOVERY_LOCK
    """
    Open only after crossover + funding + scanner + capital safety gates.

    IMPORTANT (concurrency): this function must be called WITHOUT
    holding state_lock. It takes the lock itself only for short state
    reads/writes; the network calls (set_symbol_leverage,
    place_market_order) run OUTSIDE any lock, so a slow/hanging HTTP
    request never blocks price/candle updates or stop-loss checks for
    other symbols. The position slot is reserved (in_position = True)
    under lock BEFORE the network call starts, so two callers can
    never race into opening the same symbol twice; the reservation is
    rolled back if the order actually fails.
    """

    if BOT_SHUTTING_DOWN:
        print(f"[ENTRY BLOCKED] {symbol}: bot shutdown is in progress.")
        return False

    if LIVE_MODE and ORDERS_ENABLED and RECOVERY_LOCK:
        print(f"[ENTRY BLOCKED] {symbol}: LIVE recovery/protection lock is active.")
        return False

    if LIVE_MODE and (account_balance is None or account_balance <= 0):
        print(f"[ENTRY BLOCKED] {symbol}: execution-account USDT balance is unavailable/zero.")
        return False

    blocked, funding_time, _, _ = funding_status()
    if blocked:
        print(f"[ENTRY BLOCKED] {symbol} {side}: funding blackout around {funding_time.strftime('%H:%M UTC')}")
        return False

    _fb, _fmins = symbol_funding_blocked(symbol)
    if _fb:
        print(f"[ENTRY BLOCKED] {symbol} {side}: funding in {_fmins:.1f} min (exchange timestamp)")
        return False

    _risk_ok, _risk_why = risk_entry_gate(symbol, side)
    if not _risk_ok:
        print(f"[ENTRY BLOCKED] {symbol} {side}: RISK GATE -> {_risk_why}")
        return False

    if LIVE_MODE and ORDERS_ENABLED:
        _bud_ok, _bud_bucket = api_entry_budget_ok()
        if not _bud_ok:
            print(f"[ENTRY DEFERRED] {symbol} {side}: API budget low on '{_bud_bucket}' (reserved for stops/exits)")
            return False

    allowed, scanner_score, scanner_reason = hft_entry_gate(symbol, side)
    if not allowed:
        print(f"[ENTRY BLOCKED] {symbol} {side}: HFT scanner -> {scanner_reason}")
        return False

    # AI: always compute features + score (so the model keeps learning);
    # veto only when AI_MODE is ON and enough trades have been learned.
    ai_features = None
    ai_prob = None
    try:
        ai_features = ai_build_features(symbol, side, entry)
        if ai_features is not None:
            ai_prob = ai_predict(ai_features)
    except Exception as exc:
        print(f"[AI] {symbol}: feature/score error (ignored): {exc}")

    if ai_prob is not None:
        ai_log_prediction(symbol, side, ai_features, ai_prob)

    if AI_MODE:
        ai_veto, _ai_thr, ai_reason = ai_should_veto(ai_prob)
        if ai_veto:
            ai_register_veto()
            print(f"[ENTRY BLOCKED] {symbol} {side}: AI veto -> {ai_reason}")
            return False

    # If multiple symbols cross in the same ticker cycle, only the
    # highest scanner-priority candidate gets the single position slot.
    with state_lock:
        priority_symbol = entry_priority_symbol
        priority_score = entry_priority_score
    if priority_symbol is not None and priority_symbol != symbol:
        if scanner_score < (priority_score or scanner_score):
            print(f"[ENTRY PRIORITY] {symbol} deferred: {priority_symbol} has higher scanner score {priority_score:.2f}")
            return False

    capital = effective_total_capital()

    with state_lock:
        info = dict(instrument_info.get(symbol) or {})

    min_qty = to_float(info.get("min_qty"))

    if min_qty is None or min_qty <= 0:
        print(f"[ENTRY BLOCKED] {symbol}: invalid exchange min_qty.")
        return False

    selected_leverage = None
    qty = None
    margin = None
    notional = None

    if LOW_CAPITAL_MODE:
        # C-MODE: quantity NEVER grows beyond exchange minimum.
        selected_leverage, qty, margin = select_low_capital_entry(
            symbol,
            entry,
            sl,
            side,
        )

        if selected_leverage is None:
            print(
                f"[ENTRY BLOCKED] {symbol}: min_qty={min_qty:.12g} "
                f"not affordable at 10x -> 20x -> 40x, or SL risk "
                f"failed at the candidate leverage."
            )
            return False

        notional = qty * entry

    else:
        # NORMAL MODE: preserve the existing 10x allocation behavior.
        selected_leverage = normal_entry_leverage(symbol)

        if not sl_risk_allowed(
            entry,
            sl,
            leverage=selected_leverage,
            symbol=symbol,
            side=side,
        ):
            return False

        max_entry_margin = entry_margin_allocation()
        max_total_margin = max_deployable_margin()

        with state_lock:
            deployed_margin = total_deployed_margin()

        remaining_margin = max(
            0.0,
            max_total_margin - deployed_margin
        )
        target_margin = min(
            max_entry_margin,
            remaining_margin
        )

        if capital <= 0 or target_margin <= 0:
            print(
                f"[ENTRY BLOCKED] {symbol}: no capital/deployment capacity."
            )
            return False

        target_notional = target_margin * selected_leverage
        qty = target_notional / entry

        if qty + 1e-12 < min_qty:
            print(
                f"[ENTRY BLOCKED] {symbol}: calculated qty={qty:.12g} "
                f"< exchange min_qty={min_qty:.12g}"
            )
            return False

        notional = qty * entry
        margin = notional / selected_leverage

    # Primary stop is a capital-ROI stop, not a raw price percentage.
    sl = _roi_price_for_target(
        entry, side, -PRIMARY_STOP_LOSS_ROI_PCT,
        leverage=selected_leverage, margin=margin, position_size=notional,
    )
    if sl is None or not sl_risk_allowed(
        entry, sl, leverage=selected_leverage, symbol=symbol, side=side,
    ):
        print(f"[ENTRY BLOCKED] {symbol}: unable to calculate valid {PRIMARY_STOP_LOSS_ROI_PCT:.2f}% ROI stop.")
        with state_lock:
            state = symbol_states.get(symbol)
            if state:
                state["in_position"] = False
        return False

    capital_used_pct = (
        (margin / capital * 100.0)
        if capital > 0
        else 0.0
    )

    with state_lock:
        can_afford = can_afford_new_position(
            symbol,
            entry,
            selected_leverage,
        )

    if not can_afford:
        print(
            f"[ENTRY BLOCKED] {symbol}: margin {margin:.8f} USDT "
            f"exceeds available deployment capacity "
            f"{max_deployable_margin():.8f} USDT "
            f"at {selected_leverage:g}x"
        )
        return False

    # ORDERS is an independent execution switch.
    # Account mode can remain selected while execution is OFF.
    if not ORDERS_ENABLED:
        print(f"[ENTRY BLOCKED] {symbol}: ORDERS are disabled.")
        return False

    # Reserve the position slot atomically BEFORE any network call, so
    # no other candidate can slip into this symbol (or this bot's
    # single-position slot) while the order is in flight.
    with state_lock:
        if BOT_SHUTTING_DOWN:
            return False
        state = symbol_states.get(symbol)
        if state is None or state["in_position"]:
            return False
        state["in_position"] = True
        state["position_opening_until"] = time.time() + 60.0
        _ai_mark_real_entry(symbol, side, candle_start)
        state["ai_features"] = ai_features
        state["ai_prob"] = ai_prob

    # Use the leverage selected above for this entry.
    configured_leverage_by_symbol.setdefault(symbol, None)
    if configured_leverage_by_symbol.get(symbol) != selected_leverage:
        leverage_set_symbols.discard(symbol)

    needs_leverage = LIVE_MODE and ORDERS_ENABLED and symbol not in leverage_set_symbols

    if needs_leverage:
        if set_symbol_leverage(symbol, selected_leverage):  # network call - NOT under state_lock
            with state_lock:
                leverage_set_symbols.add(symbol)
        else:
            print(f"[open_position] Could not confirm {selected_leverage:g}x leverage for {symbol}.")
            with state_lock:
                state["in_position"] = False  # release the reservation
            return False

    order_response = None
    entry_fee = 0.0
    entry_fee_base = 0.0
    entry_fee_gst = 0.0

    if LIVE_MODE and ORDERS_ENABLED:
        order_response = place_market_order(
            symbol,
            side,
            qty,
        )  # network call - NOT under state_lock

        if order_response is None:
            print(f"[open_position] Order placement FAILED for {symbol} {side}.")
            with state_lock:
                state["in_position"] = False
            return False

        # Timeout after submission is ambiguous. Never resend automatically.
        if order_response.get("_ambiguous"):
            print(
                f"[open_position] AMBIGUOUS execution for {symbol} {side}. "
                f"Starting reconciliation; no duplicate order will be sent."
            )
            with state_lock:
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return False

        entry_fee = extract_actual_fee(order_response) or 0.0
        entry_fee_base = entry_fee
        entry_fee_gst = 0.0
        if extract_actual_fee(order_response) is None:
            print(
                f"[FEE WARNING] {symbol}: exchange entry-order response "
                f"did not expose a fee field; entry fee saved as 0."
            )

        # Exchange is source of truth for actual fill.
        time.sleep(LIVE_ORDER_POLL_SECONDS)

        exchange_pos = None
        for _ in range(LIVE_ORDER_POLL_ATTEMPTS):
            exchange_pos = _live_position_snapshot(symbol)
            if exchange_pos and exchange_pos.get("qty") and exchange_pos.get("entry"):
                break
            time.sleep(LIVE_ORDER_POLL_SECONDS)

        if not exchange_pos or not exchange_pos.get("qty") or not exchange_pos.get("entry"):
            print(
                f"[open_position] Could not confirm exchange position for "
                f"{symbol}. No duplicate order will be sent."
            )
            with state_lock:
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return False

        actual_entry = exchange_pos["entry"]
        actual_qty = exchange_pos["qty"]
        actual_leverage = exchange_pos.get("leverage")
        try:
            actual_leverage = float(actual_leverage) if actual_leverage is not None else None
            if actual_leverage is not None and actual_leverage <= 0:
                actual_leverage = None
        except (TypeError, ValueError):
            actual_leverage = None

        actual_margin = exchange_pos.get("margin")
        try:
            actual_margin = float(actual_margin) if actual_margin is not None else None
            if actual_margin is not None and actual_margin <= 0:
                actual_margin = None
        except (TypeError, ValueError):
            actual_margin = None

        # Prefer exchange-reported margin/leverage. If the response omits
        # them, use the leverage we explicitly set for this entry and derive
        # margin from the confirmed filled notional so ROI protection remains calculable.
        if actual_leverage is None:
            actual_leverage = float(selected_leverage)
        if actual_margin is None and actual_leverage is not None and actual_leverage > 0:
            actual_margin = actual_entry * actual_qty / actual_leverage

        sl = _roi_price_for_target(
            actual_entry, side, -PRIMARY_STOP_LOSS_ROI_PCT,
            leverage=actual_leverage, margin=actual_margin,
            position_size=actual_entry * actual_qty,
        )
        if sl is None:
            print(f"[PROTECTION CRITICAL] {symbol}: could not calculate ROI-based primary SL.")
            with state_lock:
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return False

        with state_lock:
            state["position_side"] = side
            state["position_entry"] = actual_entry
            state["actual_entry"] = actual_entry
            state["position_sl"] = sl
            state["strategy_sl"] = sl
            state["position_qty"] = actual_qty
            state["actual_qty"] = actual_qty
            state["position_size"] = actual_entry * actual_qty
            state["position_margin"] = actual_margin
            state["actual_leverage"] = actual_leverage
            state["capital_used_pct"] = (
                actual_margin / capital * 100.0 if capital > 0 else 0.0
            )
            state["scanner_score"] = scanner_score
            state["scanner_quality"] = scanner_reason
            state["scanner_bias"] = "LONG" if side == "BUY" else "SHORT"
            state["position_open_time"] = time.time()
            state["entry_fee"] = entry_fee
            state["entry_fee_base"] = entry_fee_base
            state["entry_fee_gst"] = entry_fee_gst
            state["exchange_order_id"] = _extract_order_id(order_response)
            state["recovery_status"] = "PROTECTING"

        # Primary exchange SL is mandatory before considering the position protected.
        if not _protect_live_position(symbol, allow_emergency=True):
            print(
                f"[open_position] {symbol}: exchange SL could not be verified. "
                f"Recovery lock remains active."
            )
            return False

    else:
        if PAPER_MODE and ORDERS_ENABLED:
            entry_fee, entry_fee_base, entry_fee_gst = _paper_fee_for_notional(notional)
        else:
            entry_fee, entry_fee_base, entry_fee_gst = 0.0, 0.0, 0.0
        with state_lock:
            state["position_side"] = side
            state["position_entry"] = entry
            state["actual_entry"] = entry
            state["position_sl"] = sl
            state["strategy_sl"] = sl
            state["position_qty"] = qty
            state["actual_qty"] = qty
            state["position_size"] = notional
            state["position_margin"] = margin
            state["actual_leverage"] = selected_leverage
            state["capital_used_pct"] = capital_used_pct
            state["scanner_score"] = scanner_score
            state["scanner_quality"] = scanner_reason
            state["scanner_bias"] = "LONG" if side == "BUY" else "SHORT"
            state["position_open_time"] = time.time()
            state["entry_fee"] = entry_fee
            state["entry_fee_base"] = entry_fee_base
            state["entry_fee_gst"] = entry_fee_gst

    print()
    print("#" * 68)
    print(f">>> POSITION OPENED: {symbol} {side} <<<")
    print("#" * 68)
    with state_lock:
        _display_entry = state.get("actual_entry") or entry
        _display_sl = state.get("active_sl") or sl
        _display_qty = state.get("actual_qty") or qty
        _display_margin = state.get("position_margin") or margin
        _display_notional = state.get("position_size") or notional
        _display_leverage = state.get("actual_leverage") or TRADE_LEVERAGE

    print(f"Entry            : {_display_entry:.8f}")
    print(f"Stop Loss        : {_display_sl:.8f}")
    print(f"Quantity         : {_display_qty}")
    print(f"Margin Used      : {_display_margin:.6f} USDT")
    print(f"Capital Used     : {capital_used_pct:.3f}%")
    print(f"Notional         : {_display_notional:.6f} USDT")
    print(f"Leverage         : {_display_leverage}x")
    print(f"HFT Priority     : {scanner_score:.2f} ({scanner_reason})")
    if ai_prob is not None:
        print(f"AI Win Prob      : {ai_prob * 100:.1f}% ({'AI ON' if AI_MODE else 'advisory'})")
    if LIVE_MODE and ORDERS_ENABLED:
        print("Real Order       : YES")
    elif PAPER_MODE and ORDERS_ENABLED:
        print("Real Order       : NO (paper trade)")
    else:
        print("Real Order       : NO (orders disabled)")
    print("#" * 68)
    return True


def close_position(symbol, exit_price, reason):
    """
    IMPORTANT (concurrency): like open_position(), this must be
    called WITHOUT holding state_lock. The closing order network
    call runs outside any lock. `position_closing` is set under
    lock BEFORE that call starts (in_position stays True so no new
    entry can race in on this symbol mid-close) and is rolled back
    if the closing order actually fails, so the exit will simply be
    retried on the next price tick.
    """
    global paper_capital, trade_log
    global RECOVERY_LOCK

    if LIVE_MODE and not ORDERS_ENABLED:
        print(
            f"[close_position] {symbol}: LIVE_MODE is ON but ORDERS are OFF. "
            f"No exchange close sent; local position state and logs untouched."
        )
        return

    with state_lock:
        state = symbol_states.get(symbol)
        if state is None or not state["in_position"] or state.get("position_closing"):
            return
        side = state["position_side"]
        entry = state["position_entry"]
        qty = state.get("actual_qty") or state["position_qty"]
        size = state["position_size"]
        sl_at_open = state["active_sl"] or state["position_sl"]
        actual_entry = state.get("actual_entry") or state["position_entry"]
        actual_leverage = state.get("actual_leverage")
        try:
            actual_leverage = float(actual_leverage) if actual_leverage is not None else 0.0
        except (TypeError, ValueError):
            actual_leverage = 0.0

        capital_used_pct = state.get("capital_used_pct")
        scanner_score = state.get("scanner_score")
        scanner_quality = state.get("scanner_quality")
        open_time = state["position_open_time"]
        entry_fee = float(state.get("entry_fee") or 0.0)
        ai_features = state.get("ai_features")
        ai_prob = state.get("ai_prob")

        margin = state.get("position_margin")
        try:
            margin = float(margin) if margin is not None else 0.0
        except (TypeError, ValueError):
            margin = 0.0

        if margin <= 0 and size and actual_leverage > 0:
            margin = size / actual_leverage
        state["position_closing"] = True  # reserve the close

    order_response = None
    exit_fee = 0.0

    if LIVE_MODE and ORDERS_ENABLED:
        opposite_side = "SELL" if side == "BUY" else "BUY"
        order_response = place_market_order(symbol, opposite_side, qty, reduce_only=True)  # network call - NOT under state_lock

        # A timeout/ambiguous response means the exchange may have accepted
        # the close. Never resend the close and never cancel the protective
        # STOP until the exchange position is explicitly reconciled.
        if order_response is None or (
            isinstance(order_response, dict) and order_response.get("_ambiguous")
        ):
            print(
                f"[close_position] AMBIGUOUS/FAILED close for {symbol}. "
                f"No retry order and protective STOP will remain active."
            )
            with state_lock:
                state["position_closing"] = False
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return

        actual_fee = extract_actual_fee(order_response)
        if actual_fee is not None:
            exit_fee = actual_fee
        else:
            print(
                f"[FEE WARNING] {symbol}: exchange close-order response "
                f"did not expose a fee field; exit fee saved as 0. "
                f"No estimated fee used."
            )

        # Exchange-reported average execution price is the authoritative
        # live exit price. The strategy trigger price is only the requested
        # market-exit reference and must not be used for final accounting.
        exchange_exit_price = extract_actual_execution_price(order_response)

        if exchange_exit_price is not None:
            exit_price = exchange_exit_price
            print(
                f"[LIVE FILL] {symbol}: actual exit fill "
                f"{exit_price:.8f}"
            )
        else:
            print(
                f"[FILL WARNING] {symbol}: close response did not expose "
                f"avg_execution_price. Using requested exit price "
                f"{exit_price:.8f}; no estimated fill will be invented."
            )

        # Exchange is the source of truth. Confirm that the position is
        # actually gone before removing its protective STOP.
        close_verified = False
        remaining_qty = None

        for _ in range(LIVE_ORDER_POLL_ATTEMPTS):
            raw_after = get_live_positions(symbol)

            if raw_after is None:
                # API failure/unknown state. Do not assume the position closed.
                time.sleep(LIVE_ORDER_POLL_SECONDS)
                continue

            pos_after = _extract_position_from_payload(raw_after, symbol)

            if pos_after is None:
                close_verified = True
                break

            try:
                remaining_qty = float(pos_after.get("qty") or 0.0)
            except Exception:
                remaining_qty = None

            if remaining_qty is not None and remaining_qty <= 0:
                close_verified = True
                break

            # Partial close or residual position. Keep the protective STOP.
            if remaining_qty is not None and remaining_qty > 0:
                print(
                    f"[close_position] PARTIAL/RESIDUAL position remains for "
                    f"{symbol}: qty={remaining_qty}. Protective STOP retained."
                )
                with state_lock:
                    state["actual_qty"] = remaining_qty
                    state["position_qty"] = remaining_qty
                    state["position_closing"] = False
                    state["recovery_status"] = "RECOVERING"
                    state["sl_status"] = "VERIFIED"
                RECOVERY_LOCK = True
                return

            time.sleep(LIVE_ORDER_POLL_SECONDS)

        if not close_verified:
            print(
                f"[close_position] CLOSE NOT VERIFIED for {symbol}. "
                f"Protective STOP remains active; recovery lock enabled."
            )
            with state_lock:
                state["position_closing"] = False
                state["recovery_status"] = "RECOVERING"
                state["sl_status"] = "UNKNOWN"
            RECOVERY_LOCK = True
            return

        print(
            f"[close_position] EXCHANGE POSITION CLOSED VERIFIED for {symbol}."
        )

        # Only after zero position is verified, remove the old protective STOP.
        old_sl_order_id = state.get("sl_order_id")
        if old_sl_order_id:
            cancel_result = cancel_order(old_sl_order_id)
            if cancel_result is None:
                print(
                    f"[PROTECTION WARNING] {symbol}: could not cancel "
                    f"old STOP_MARKET order {old_sl_order_id} after verified close. "
                    f"Open-order reconciliation will handle it."
                )
            else:
                print(
                    f"[PROTECTION] {symbol}: old STOP_MARKET "
                    f"{old_sl_order_id} cancellation requested after verified close."
                )

    entry_fee_base = float(state.get("entry_fee_base") or 0.0)
    entry_fee_gst = float(state.get("entry_fee_gst") or 0.0)
    exit_fee_base = 0.0
    exit_fee_gst = 0.0
    if LIVE_MODE and ORDERS_ENABLED:
        fee = entry_fee + exit_fee
    elif PAPER_MODE:
        exit_notional = abs(float(qty or 0.0) * float(exit_price or 0.0))
        exit_fee, exit_fee_base, exit_fee_gst = _paper_fee_for_notional(exit_notional)
        fee = float(entry_fee or 0.0) + exit_fee
    else:
        fee = 0.0

    entry = actual_entry
    raw_pnl_pct = calculate_raw_pnl_pct(entry, exit_price, side)
    pnl_usdt = size * raw_pnl_pct / 100.0 if size else 0.0
    margin_roi_pct = (pnl_usdt / margin * 100.0) if margin else 0.0

    # For an existing live position, ROI is based on actual position
    # margin. Do not re-apply configured entry leverage.
    pnl_pct = margin_roi_pct
    net_pnl_usdt = pnl_usdt - fee
    timestamp = time.time()

    record = {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "exit": exit_price,
        "sl": sl_at_open,
        "size": size,
        "margin": margin,
        "capital_used_pct": capital_used_pct,
        "leverage": actual_leverage,
        "actual_entry": actual_entry,
        "actual_qty": qty,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "pnl_usdt": pnl_usdt,
        "fee": fee,
        "entry_fee": float(entry_fee or 0.0),
        "entry_fee_base": entry_fee_base,
        "entry_fee_gst": entry_fee_gst,
        "exit_fee": float(exit_fee or 0.0),
        "exit_fee_base": exit_fee_base,
        "exit_fee_gst": exit_fee_gst,
        "net_pnl_usdt": net_pnl_usdt,
        "margin_roi_pct": margin_roi_pct,
        "scanner_score": scanner_score,
        "scanner_quality": scanner_quality,
        "opened_at": open_time,
        "closed_at": timestamp,
        "mode": "live" if LIVE_MODE else ("paper" if PAPER_MODE else "none"),
        "ai_features": ai_features,
        "ai_prob": ai_prob,
    }

    trade_log.append(record)

    try:
        ai_learn_from_record(record)
    except Exception as exc:
        print(f"[AI] learning error (ignored): {exc}")

    if LIVE_MODE:
        live_trade_log.append(record)
        _atomic_write_json(LIVE_LOG_FILE, live_trade_log)  # file I/O - NOT under state_lock
    elif PAPER_MODE:
        paper_capital = paper_capital + net_pnl_usdt
        paper_trade_log.append({**record, "resulting_capital": paper_capital})
        _atomic_write_json(PAPER_LOG_FILE, paper_trade_log)  # file I/O - NOT under state_lock
        save_paper_capital()

    print()
    print("#" * 68)
    print(f">>> POSITION CLOSED: {symbol} {side} <<<")
    print("#" * 68)
    print(f"Entry            : {entry:.8f}")
    print(f"Exit             : {exit_price:.8f}")
    print(f"Reason           : {reason}")
    print(f"Price Move       : {raw_pnl_pct:+.3f}%")
    print(f"Gross PnL        : {pnl_usdt:+.6f} USDT")
    print(f"Entry Fee        : {float(entry_fee or 0.0):.6f} USDT")
    print(f"Exit Fee         : {float(exit_fee or 0.0):.6f} USDT")
    if PAPER_MODE:
        print(f"Entry Fee GST    : {entry_fee_gst:.6f} USDT")
        print(f"Exit Fee GST     : {exit_fee_gst:.6f} USDT")
    print(f"Total Fees       : {fee:.6f} USDT")
    print(f"Net PnL          : {net_pnl_usdt:+.6f} USDT")
    print(f"Margin ROI       : {margin_roi_pct:+.3f}%")
    print(f"Position ROI     : {pnl_pct:+.3f}%"
          + (f" @ {actual_leverage:g}x" if actual_leverage > 0 else ""))
    if PAPER_MODE:
        print(f"Paper Capital    : {paper_capital:.6f} USDT")
    print("#" * 68)

    _clear_local_position_state(
        symbol,
        recovery_status="CLOSED",
    )


def check_position_exit(symbol, price, current_candle_now=None, previous_candle_now=None):
    """Exit an open position only when its current ROI-based SL is hit."""
    state = symbol_states[symbol]
    side = state.get("position_side")
    sl = state.get("position_sl")
    if price is None or sl is None:
        return False, None, None
    try:
        price = float(price)
        sl = float(sl)
    except (TypeError, ValueError):
        return False, None, None
    if side == "BUY" and price <= sl:
        return True, price, "STOP LOSS HIT"
    if side == "SELL" and price >= sl:
        return True, price, "STOP LOSS HIT"
    return False, None, None


# ============================================================
# TRAILING STOP HELPERS
# ============================================================

def _roi_price_for_target(entry, side, target_roi_pct, leverage=None,
                          margin=None, position_size=None):
    """Convert a capital/ROI target into the corresponding market price."""
    try:
        entry = float(entry)
        target_roi_pct = float(target_roi_pct)
    except (TypeError, ValueError):
        return None

    if entry <= 0:
        return None

    # Prefer actual margin + notional because that is the same basis used
    # by calculate_leveraged_pnl_pct(). Fall back to actual leverage.
    effective_leverage = None
    try:
        if margin is not None and position_size is not None:
            margin_f = float(margin)
            size_f = float(position_size)
            if margin_f > 0 and size_f > 0:
                effective_leverage = size_f / margin_f
    except (TypeError, ValueError, ZeroDivisionError):
        effective_leverage = None

    if effective_leverage is None:
        try:
            lev = float(leverage) if leverage is not None else 0.0
            if lev > 0:
                effective_leverage = lev
        except (TypeError, ValueError):
            pass

    if effective_leverage is None or effective_leverage <= 0:
        return None

    raw_move_pct = target_roi_pct / effective_leverage

    if side == "BUY":
        return entry * (1.0 + raw_move_pct / 100.0)
    if side == "SELL":
        return entry * (1.0 - raw_move_pct / 100.0)

    return None


def _calculate_trailing_candidate(side, entry, current_price, current_sl,
                                  trailing_peak=None, leverage=None,
                                  margin=None, position_size=None,
                                  trailing_armed=False):
    """ROI trailing: activate at +7.25%, lock +5.30%, then trail only the excess ROI by 20%."""
    if not TRAILING_SL_ENABLED:
        return None
    try:
        entry = float(entry)
        price = float(current_price)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or price <= 0 or side not in ("BUY", "SELL"):
        return None

    roi = calculate_leveraged_pnl_pct(
        entry, price, side, leverage=leverage, margin=margin,
        position_size=position_size,
    )
    raw_now = calculate_raw_pnl_pct(entry, price, side)
    if roi == 0.0 and abs(raw_now) > 1e-12:
        return None

    previous_peak = None
    try:
        previous_peak = float(trailing_peak) if trailing_peak is not None else None
    except (TypeError, ValueError):
        previous_peak = None
    if side == "BUY":
        new_peak = max(price, previous_peak) if previous_peak is not None else price
    else:
        new_peak = min(price, previous_peak) if previous_peak is not None else price

    peak_roi = calculate_leveraged_pnl_pct(
        entry, new_peak, side, leverage=leverage, margin=margin,
        position_size=position_size,
    )
    if peak_roi < TRAILING_ACTIVATION_PCT:
        return None

    if not trailing_armed:
        # On the first +7.25% activation, lock exactly +5.30% ROI.
        target_roi = TRAILING_INITIAL_LOCK_ROI_PCT
    else:
        # Thereafter retrace 20% of profit accumulated ABOVE +7.25% ROI.
        target_roi = TRAILING_INITIAL_LOCK_ROI_PCT + (
            peak_roi - TRAILING_ACTIVATION_PCT
        ) * (1.0 - TRAILING_RETRACEMENT_PCT / 100.0)
    stage = "ACTIVE"

    trigger = _roi_price_for_target(
        entry, side, target_roi, leverage=leverage, margin=margin,
        position_size=position_size,
    )
    if trigger is None:
        return None

    if current_sl is not None:
        try:
            current_sl = float(current_sl)
            if side == "BUY" and trigger <= current_sl:
                return None
            if side == "SELL" and trigger >= current_sl:
                return None
        except (TypeError, ValueError):
            pass

    return {
        "side": side,
        "peak": new_peak,
        "peak_roi": peak_roi,
        "trigger": trigger,
        "stage": stage,
        "armed": True,
        "roi": roi,
        "target_roi": target_roi,
    }


def _replace_live_trailing_stop(symbol, position_side, candidate,
                                quantity, old_sl_order_id, current_sl):
    """Per-symbol re-entrancy guard around the STOP replacement."""
    with state_lock:
        _st = symbol_states.get(symbol)
        if _st is None or _st.get("trailing_replacing"):
            return None
        _st["trailing_replacing"] = True

    try:
        return _replace_live_trailing_stop_inner(
            symbol, position_side, candidate,
            quantity, old_sl_order_id, current_sl,
        )
    finally:
        with state_lock:
            _st = symbol_states.get(symbol)
            if _st is not None:
                _st["trailing_replacing"] = False


def _replace_live_trailing_stop_inner(symbol, position_side, candidate,
                                      quantity, old_sl_order_id, current_sl):
    """
    LIVE trailing STOP replacement.

    CoinSwitch currently rejects a second STOP for the same position.
    Therefore replacement must be:

        1. Confirm/identify the existing STOP when possible.
        2. Cancel the old STOP.
        3. Immediately place the tighter STOP.
        4. Verify the new exchange-side STOP.
        5. Return verified protection information.

    Local trailing state is committed only by _commit_trailing_state()
    after this function returns a verified result.

    If replacement fails after cancellation, immediately attempt to
    restore the previous STOP. The caller must not commit the new
    trailing state unless the new STOP is verified.
    """
    global RECOVERY_LOCK
    if not LIVE_MODE or not ORDERS_ENABLED:
        return None

    if candidate is None or quantity is None:
        return None

    try:
        candidate = float(candidate)
        current_sl = float(current_sl) if current_sl is not None else None
    except (TypeError, ValueError):
        return None

    if current_sl is not None:
        if position_side == "BUY" and candidate <= current_sl:
            return None
        if position_side == "SELL" and candidate >= current_sl:
            return None

    # Prevent the same failed replacement from hammering the API on
    # every ticker tick. A successful replacement clears this state
    # through the normal trailing-state commit.
    now = time.time()

    try:
        with state_lock:
            state = symbol_states.get(symbol)
            retry_block_until = (
                float(state.get("trailing_replace_block_until") or 0.0)
                if state
                else 0.0
            )
    except Exception:
        retry_block_until = 0.0

    if now < retry_block_until:
        return None

    print(
        f"[TRAILING] {symbol} {position_side}: "
        f"replacing STOP with tighter trigger {candidate:.8f}"
    )

    # If local order ID is missing, first inspect open orders and find
    # the currently active protective STOP. Never blindly create a
    # second STOP when one may already exist.
    existing_order_id = old_sl_order_id
    existing_trigger = current_sl

    if not existing_order_id:
        open_payload = get_live_open_orders(symbol=symbol, limit=50)

        if open_payload is None:
            print(
                f"[TRAILING WARNING] {symbol}: unable to inspect open orders. "
                f"No new STOP will be placed."
            )
            try:
                with state_lock:
                    state = symbol_states.get(symbol)
                    if state:
                        state["trailing_replace_block_until"] = 0.0
            except Exception:
                pass
            return None

        close_side = "SELL" if position_side == "BUY" else "BUY"

        try:
            with state_lock:
                state = symbol_states.get(symbol)
                position_entry = (
                    (state.get("actual_entry") or state.get("position_entry"))
                    if state else None
                )
        except Exception:
            position_entry = None

        for order in _extract_order_rows(open_payload):
            if not _order_is_stop_protection(
                order,
                symbol,
                close_side,
                candidate,
                position_entry=position_entry,
            ):
                # The existing STOP may be looser than the requested
                # trailing trigger. Check it separately so it can be
                # replaced safely.
                osymbol = str(
                    order.get("symbol")
                    or order.get("instrument")
                    or ""
                ).upper()

                otype = str(
                    order.get("order_type")
                    or order.get("orderType")
                    or ""
                ).upper()

                oside = str(order.get("side") or "").upper()

                reduce_only = order.get("reduce_only")
                if reduce_only is None:
                    reduce_only = order.get("reduceOnly")

                actual_trigger = _first_number(
                    order,
                    (
                        "trigger_price",
                        "triggerPrice",
                        "stop_price",
                        "stopPrice",
                    ),
                )

                if (
                    (not osymbol or osymbol == str(symbol).upper())
                    and "STOP" in otype
                    and (not oside or oside == close_side)
                    and reduce_only is not False
                    and actual_trigger is not None
                ):
                    existing_order_id = _extract_order_id(order)
                    existing_trigger = float(actual_trigger)
                    break

            else:
                existing_order_id = _extract_order_id(order)
                existing_trigger = _first_number(
                    order,
                    (
                        "trigger_price",
                        "triggerPrice",
                        "stop_price",
                        "stopPrice",
                    ),
                )
                break

    # ------------------------------------------------------------
    # Existing STOP must be removed before CoinSwitch will accept
    # another STOP for this position.
    # ------------------------------------------------------------
    if existing_order_id:
        print(
            f"[TRAILING] {symbol}: cancelling existing STOP "
            f"{existing_order_id} @ "
            f"{float(existing_trigger):.8f}"
            if existing_trigger is not None
            else
            f"[TRAILING] {symbol}: cancelling existing STOP "
            f"{existing_order_id}"
        )

        cancel_result = cancel_order(existing_order_id)

        if cancel_result is None:
            print(
                f"[TRAILING WARNING] {symbol}: existing STOP "
                f"{existing_order_id} could not be cancelled. "
                f"New STOP NOT placed."
            )
            try:
                with state_lock:
                    state = symbol_states.get(symbol)
                    if state:
                        state["trailing_replace_block_until"] = 0.0
            except Exception:
                pass
            return None

        with state_lock:
            state = symbol_states.get(symbol)
            if state:
                state["sl_status"] = "UNKNOWN"
                state["recovery_status"] = "RECOVERING"

    else:
        print(
            f"[TRAILING WARNING] {symbol}: no existing STOP order ID "
            f"could be confirmed before replacement."
        )
        try:
            with state_lock:
                state = symbol_states.get(symbol)
                if state:
                    state["trailing_replace_block_until"] = 0.0
        except Exception:
            pass
        return None

    # ------------------------------------------------------------
    # Immediately place the tighter replacement.
    # ------------------------------------------------------------
    new_order = place_exchange_stop_loss(
        symbol,
        position_side,
        candidate,
        quantity,
    )

    if new_order is None or new_order.get("ambiguous"):
        print(
            f"[TRAILING WARNING] {symbol}: new trailing STOP "
            f"could not be confirmed after old STOP cancellation. "
            f"Attempting to restore previous STOP."
        )

        # Best-effort restoration of the old exchange protection.
        if existing_trigger is not None:
            restored = place_exchange_stop_loss(
                symbol,
                position_side,
                existing_trigger,
                quantity,
            )

            if restored is not None and not restored.get("ambiguous"):
                restore_verify = verify_exchange_stop_loss(
                    symbol,
                    position_side,
                    existing_trigger,
                )

                if restore_verify and restore_verify.get("verified"):
                    restored_id = (
                        restore_verify.get("order_id")
                        or restored.get("order_id")
                    )
                    with state_lock:
                        state = symbol_states.get(symbol)
                        if state:
                            state["sl_order_id"] = restored_id
                            state["sl_status"] = "VERIFIED"
                            state["recovery_status"] = "PROTECTED"
                    print(
                        f"[TRAILING RECOVERY] {symbol}: previous STOP "
                        f"restored and exchange-verified."
                    )
                else:
                    print(
                        f"[TRAILING CRITICAL] {symbol}: previous STOP "
                        f"restore could not be verified. "
                        f"Recovery lock must remain active."
                    )
            else:
                print(
                    f"[TRAILING CRITICAL] {symbol}: previous STOP "
                    f"restore placement failed."
                )

        try:
            with state_lock:
                state = symbol_states.get(symbol)
                if state:
                    state["trailing_replace_block_until"] = 0.0
                    if state.get("sl_status") != "VERIFIED":
                        state["sl_status"] = "UNKNOWN"
                        state["recovery_status"] = "RECOVERING"
                        RECOVERY_LOCK = True
        except Exception:
            pass

        return None

    # ------------------------------------------------------------
    # Verify the newly placed STOP before committing local state.
    # ------------------------------------------------------------
    verification = verify_exchange_stop_loss(
        symbol,
        position_side,
        candidate,
    )

    if verification is None or not verification.get("verified"):
        print(
            f"[TRAILING WARNING] {symbol}: new trailing STOP "
            f"failed exchange verification. Local trailing state NOT committed. "
            f"Attempting to restore previous STOP."
        )

        if existing_trigger is not None:
            restored = place_exchange_stop_loss(
                symbol,
                position_side,
                existing_trigger,
                quantity,
            )

            if restored is not None and not restored.get("ambiguous"):
                restore_verify = verify_exchange_stop_loss(
                    symbol,
                    position_side,
                    existing_trigger,
                )

                if restore_verify and restore_verify.get("verified"):
                    restored_id = (
                        restore_verify.get("order_id")
                        or restored.get("order_id")
                    )
                    with state_lock:
                        state = symbol_states.get(symbol)
                        if state:
                            state["sl_order_id"] = restored_id
                            state["sl_status"] = "VERIFIED"
                            state["recovery_status"] = "PROTECTED"
                    print(
                        f"[TRAILING RECOVERY] {symbol}: previous STOP "
                        f"restored and exchange-verified."
                    )
                else:
                    print(
                        f"[TRAILING CRITICAL] {symbol}: previous STOP "
                        f"restore could not be verified."
                    )

        try:
            with state_lock:
                state = symbol_states.get(symbol)
                if state:
                    state["trailing_replace_block_until"] = 0.0
                    if state.get("sl_status") != "VERIFIED":
                        state["sl_status"] = "UNKNOWN"
                        state["recovery_status"] = "RECOVERING"
                        RECOVERY_LOCK = True
        except Exception:
            pass

        return None

    verified_trigger = verification.get("trigger_price")

    try:
        verified_trigger = float(verified_trigger)
    except (TypeError, ValueError):
        verified_trigger = None

    if verified_trigger is None:
        print(
            f"[TRAILING WARNING] {symbol}: verified STOP has no valid "
            f"trigger price. Local trailing state NOT committed; protection recovery required."
        )
        with state_lock:
            state = symbol_states.get(symbol)
            if state:
                state["sl_status"] = "UNKNOWN"
                state["recovery_status"] = "RECOVERING"
        RECOVERY_LOCK = True
        return None

    if current_sl is not None:
        if position_side == "BUY" and verified_trigger <= current_sl:
            print(
                f"[TRAILING WARNING] {symbol}: verified trigger "
                f"{verified_trigger:.8f} is not tighter than current "
                f"SL {current_sl:.8f}. Local state NOT committed."
            )
            return None

        if position_side == "SELL" and verified_trigger >= current_sl:
            print(
                f"[TRAILING WARNING] {symbol}: verified trigger "
                f"{verified_trigger:.8f} is not tighter than current "
                f"SL {current_sl:.8f}. Local state NOT committed."
            )
            return None

    placed_order_id = new_order.get("order_id")
    verified_order_id = verification.get("order_id")

    if verified_order_id is None:
        print(
            f"[TRAILING WARNING] {symbol}: exchange verification "
            f"did not return the new STOP order ID. "
            f"Local trailing state NOT committed."
        )
        return None

    if (
        placed_order_id is not None
        and str(verified_order_id) != str(placed_order_id)
    ):
        print(
            f"[TRAILING WARNING] {symbol}: verified STOP order "
            f"{verified_order_id} does not match newly placed STOP "
            f"{placed_order_id}. Local state NOT committed."
        )
        return None

    new_order_id = verified_order_id or placed_order_id

    try:
        with state_lock:
            state = symbol_states.get(symbol)
            if state:
                state["trailing_replace_block_until"] = 0.0
                state["sl_order_id"] = new_order_id
                state["sl_status"] = "VERIFIED"
                state["recovery_status"] = "PROTECTED"
    except Exception:
        pass

    print(
        f"[TRAILING VERIFIED] {symbol} {position_side}: "
        f"STOP={verified_trigger:.8f} order_id={new_order_id}"
    )

    return {
        "verified": True,
        "trigger_price": verified_trigger,
        "order_id": new_order_id,
    }


def _commit_trailing_state(symbol, trailing_info, protection=None):
    """
    Commit trailing state only after the new LIVE STOP has been
    verified, or immediately for PAPER/local trailing.
    """
    if trailing_info is None:
        return

    with state_lock:
        state = symbol_states.get(symbol)

        if not state or not state.get("in_position"):
            return

        trigger = float(trailing_info["trigger"])
        peak = float(trailing_info["peak"])

        state["trailing_armed"] = bool(trailing_info.get("armed", True))
        state["trailing_stage"] = str(trailing_info.get("stage") or "ACTIVE")
        state["trailing_active"] = state["trailing_stage"] == "ACTIVE"
        state["trailing_peak"] = peak
        state["trailing_sl"] = trigger
        state["position_sl"] = trigger
        state["strategy_sl"] = trigger
        state["active_sl"] = trigger
        state["protection_type"] = (
            "TRAILING" if state["trailing_active"] else "TRAILING_ARMED"
        )

        if protection is not None:
            verified_trigger = protection.get("trigger_price")
            if verified_trigger is not None:
                trigger = float(verified_trigger)
                state["trailing_sl"] = trigger
                state["position_sl"] = trigger
                state["strategy_sl"] = trigger
                state["active_sl"] = trigger

            state["sl_order_id"] = protection.get("order_id")
            state["trailing_last_order_id"] = protection.get("order_id")
            state["sl_status"] = "VERIFIED"
            state["protection_verified_at"] = time.time()
            state["protection_last_check"] = time.time()

        entry = state.get("actual_entry") or state.get("position_entry")
        qty = state.get("actual_qty") or state.get("position_qty")
        margin = state.get("position_margin")

        if entry:
            state["sl_distance_pct"] = (
                abs(float(entry) - float(state["position_sl"]))
                / float(entry) * 100.0
            )

            # Keep the dashboard/risk state synchronized with the
            # newly committed trailing protection.
            state["sl_capital_risk_pct"] = _actual_sl_risk_pct(
                float(entry),
                float(state["position_sl"]),
                qty,
                margin,
            )

        print(
            f"[TRAILING ACTIVE] {symbol} {state.get('position_side')}: "
            f"peak={peak:.8f} SL={float(state['position_sl']):.8f}"
        )


# ============================================================
# CORE PRICE-DRIVEN LOGIC (per symbol)
# ============================================================

def handle_new_price(symbol, price, bid=None, ask=None, mark=None, trade_time=None):
    """
    Updates live price state for `symbol`, manages its open
    position's exit, and looks for fresh entry signals. Called
    once per REST ticker poll for every tracked symbol.
    """

    try:

        if price is None:
            return

        with state_lock:

            if symbol not in symbol_states:
                return

            state = symbol_states[symbol]

            old_price = state["previous_live_price"]

            state["live_price"] = price

            if bid is not None:
                state["bid_price"] = bid
            if ask is not None:
                state["ask_price"] = ask
            if mark is not None:
                state["mark_price"] = mark
            if trade_time is not None:
                state["last_trade_time"] = trade_time

            current = (
                dict(state["current_candle"])
                if state["current_candle"] is not None
                else None
            )

            previous = (
                dict(state["previous_closed_candle"])
                if state["previous_closed_candle"] is not None
                else None
            )

            current_signal_candle = state["signal_candle_start"]
            already_triggered = state["signal_triggered"]

            # --------------------------------------------------
            # STEP 1: If a position is open on this symbol, ONLY
            # manage the exit. No new entries evaluated for it.
            # The actual close_position() call (which can place a
            # real closing order over the network) happens AFTER
            # this lock is released - see below.
            # --------------------------------------------------

            in_position_now = state["in_position"]
            pending_exit = None
            trailing_action = None

            if in_position_now:

                # --------------------------------------------------
                # TRAILING STOP
                # --------------------------------------------------
                # Calculate only under state_lock. No network call
                # is allowed here.
                #
                # If an opposite exit is triggered on this same tick,
                # trailing replacement is skipped because the position
                # is already scheduled to close.
                # --------------------------------------------------

                side_now = state.get("position_side")
                entry_now = state.get("actual_entry") or state.get("position_entry")
                current_sl_now = state.get("position_sl")
                trailing_peak_now = state.get("trailing_peak")
                trailing_armed_now = bool(state.get("trailing_armed"))
                qty_now = state.get("actual_qty") or state.get("position_qty")
                margin_now = state.get("position_margin")
                leverage_now = state.get("actual_leverage")
                position_size_now = state.get("position_size")
                old_sl_order_id_now = state.get("sl_order_id")

                trailing_candidate = _calculate_trailing_candidate(
                    side_now,
                    entry_now,
                    price,
                    current_sl_now,
                    trailing_peak_now,
                    leverage=leverage_now,
                    margin=margin_now,
                    position_size=position_size_now,
                    trailing_armed=trailing_armed_now,
                )

                if trailing_candidate is not None:

                    if LIVE_MODE and ORDERS_ENABLED:

                        trailing_action = {
                            "candidate": trailing_candidate,
                            "quantity": qty_now,
                            "old_sl_order_id": old_sl_order_id_now,
                            "current_sl": current_sl_now,
                        }

                    else:
                        # PAPER/local trailing needs no exchange call.
                        state["trailing_armed"] = bool(trailing_candidate.get("armed", True))
                        state["trailing_stage"] = str(trailing_candidate.get("stage") or "ACTIVE")
                        state["trailing_active"] = state["trailing_stage"] == "ACTIVE"
                        state["trailing_peak"] = trailing_candidate["peak"]
                        state["trailing_sl"] = trailing_candidate["trigger"]
                        state["position_sl"] = trailing_candidate["trigger"]
                        state["strategy_sl"] = trailing_candidate["trigger"]
                        state["active_sl"] = trailing_candidate["trigger"]
                        state["protection_type"] = (
                            "TRAILING" if state["trailing_active"] else "TRAILING_ARMED"
                        )

                should_exit, exit_price, reason = check_position_exit(
                    symbol, price, current, previous
                )

                if should_exit:
                    pending_exit = (exit_price, reason)
                    trailing_action = None

            state["previous_live_price"] = price

            # --------------------------------------------------
            # STEP 2: Only look for NEW entries if this symbol is
            # still part of the active watchlist AND not already
            # in a position.
            # --------------------------------------------------

            symbol_is_watched = (not in_position_now) and (symbol in watchlist)

        # Lock released above. Close the position (if flagged) OUTSIDE
        # the lock - close_position() places the real closing order
        # over the network itself and manages its own locking, so
        # this never blocks other symbols' price handling.
        if pending_exit is not None:
            close_position(symbol, pending_exit[0], pending_exit[1])

        # --------------------------------------------------------
        # LIVE trailing STOP replacement.
        #
        # Network calls are deliberately outside state_lock:
        #   1. cancel the old STOP
        #   2. immediately place the tighter STOP
        #   3. verify the replacement (restore old STOP on failure)
        #   4. commit local trailing state only after verification
        #
        # If verification fails, local trailing state is untouched.
        # --------------------------------------------------------

        if (
            in_position_now
            and pending_exit is None
            and trailing_action is not None
            and LIVE_MODE
            and ORDERS_ENABLED
        ):

            live_trailing = _replace_live_trailing_stop(
                symbol,
                trailing_action["candidate"]["side"],
                trailing_action["candidate"]["trigger"],
                trailing_action["quantity"],
                trailing_action["old_sl_order_id"],
                trailing_action["current_sl"],
            )

            if live_trailing is None:
                # Keep the observed favourable peak even if replacement failed;
                # the next price tick immediately recalculates the proper ROI
                # retracement instead of forgetting a profit peak.
                with state_lock:
                    state = symbol_states.get(symbol)
                    candidate_peak = trailing_action["candidate"].get("peak")
                    if state and candidate_peak is not None:
                        old_peak = state.get("trailing_peak")
                        try:
                            old_peak = float(old_peak) if old_peak is not None else None
                            candidate_peak = float(candidate_peak)
                            side_for_peak = trailing_action["candidate"].get("side")
                            if old_peak is None:
                                state["trailing_peak"] = candidate_peak
                            elif side_for_peak == "BUY":
                                state["trailing_peak"] = max(old_peak, candidate_peak)
                            else:
                                state["trailing_peak"] = min(old_peak, candidate_peak)
                        except (TypeError, ValueError):
                            pass

            if live_trailing is not None:

                committed_trailing = dict(trailing_action["candidate"])

                # The exchange-verified trigger is the source of truth
                # for the committed local protection level.
                committed_trailing["trigger"] = live_trailing["trigger_price"]

                _commit_trailing_state(
                    symbol,
                    committed_trailing,
                    protection=live_trailing,
                )

        if in_position_now:
            return

        # ----------------------------------------------------
        # Required data for cross detection
        # ----------------------------------------------------

        if (
            not symbol_is_watched
            or current is None
            or previous is None
            or old_price is None
        ):

            with state_lock:
                symbol_states[symbol]["previous_live_price"] = price

            return

        current_start = current["start_time"]
        if current_start == previous.get("start_time"):
            # An explicit candle-close websocket event may briefly leave
            # current/previous pointing at the same candle. Never use it
            # as its own crossover reference.
            with state_lock:
                symbol_states[symbol]["previous_live_price"] = price
            return

        # ----------------------------------------------------
        # Only one signal per current 5-minute candle
        # ----------------------------------------------------

        if (
            already_triggered
            and current_signal_candle == current_start
        ):

            with state_lock:
                symbol_states[symbol]["previous_live_price"] = price

            return

        # ----------------------------------------------------
        # ENTRY: live price must CROSS the previous candle OPEN.
        # BUY requires the previous three completed candles RED;
        # SELL requires the previous three completed candles GREEN.
        # The cross is detected from previous live tick -> current tick,
        # so merely being beyond the level does not create a late signal.
        # ----------------------------------------------------

        previous_open = float(previous["open"])
        current_open = float(current["open"])
        with state_lock:
            closed3 = _last_closed_candles(symbol_states[symbol], 3)

        colours = []
        for candle in closed3:
            try:
                co, cc = float(candle["open"]), float(candle["close"])
                colours.append("GREEN" if cc > co else ("RED" if cc < co else "DOJI"))
            except (KeyError, TypeError, ValueError):
                colours.append("DOJI")

        crossed_up = (
            old_price is not None and current_open <= previous_open
            and float(old_price) <= previous_open < float(price)
        )
        crossed_down = (
            old_price is not None and current_open >= previous_open
            and float(old_price) >= previous_open > float(price)
        )

        if crossed_up and len(colours) == 3 and all(c == "RED" for c in colours):
            side = "BUY"
        elif crossed_down and len(colours) == 3 and all(c == "GREEN" for c in colours):
            side = "SELL"
        else:
            side = None

        if side is not None and REQUIRE_CONTIGUOUS_CANDLES and not _setup_candles_ok(current_start, closed3):
            _note_gap_skip(symbol, current_start)
            side = None

        if side is not None:
            # AI v3: remember this setup (passive simulation, never trades).
            try:
                ai_record_setup(symbol, side, float(price), current_start)
            except Exception as _exc:
                print(f"[AI] setup record error (ignored): {_exc}")
            trigger = float(price)
            stop_loss = _roi_price_for_target(
                trigger, side, -PRIMARY_STOP_LOSS_ROI_PCT,
                leverage=normal_entry_leverage(symbol),
            )
            if stop_loss is None:
                with state_lock:
                    symbol_states[symbol]["previous_live_price"] = price
                return
            should_attempt_open = False

            with state_lock:
                state = symbol_states[symbol]

                can_open = can_afford_new_position(symbol, price)
                risk_allowed = sl_risk_allowed(
                    trigger,
                    stop_loss,
                    symbol=symbol,
                    side=side,
                )

                if (
                    (
                        not state["signal_triggered"]
                        or state["signal_candle_start"] != current_start
                    )
                    and not state["in_position"]
                    and can_open
                    and risk_allowed
                ):
                    state["signal"] = side
                    state["entry_price"] = trigger
                    state["stop_loss"] = stop_loss
                    state["signal_candle_start"] = current_start
                    should_attempt_open = True

            if should_attempt_open:
                print()
                print("=" * 68)
                print(f">>> {side} SIGNAL TRIGGERED: {symbol} <<<")
                print("=" * 68)
                print(f"Previous 3 Candles : {','.join(colours)}")
                print(f"Previous OPEN      : {previous_open:.8f}")
                print(f"Current OPEN       : {current_open:.8f}")
                print(f"Live Price         : {price:.8f}")
                print(f"Entry              : {trigger:.8f}")
                print(f"Stop Loss          : {stop_loss:.8f}")
                raw_risk = abs(trigger - stop_loss) / trigger * 100.0
                print(f"SL Price Risk      : {raw_risk:.3f}%")
                print(f"SL Capital Risk    : {raw_risk * TRADE_LEVERAGE:.3f}% @ configured {TRADE_LEVERAGE}x")
                print(f"Leverage           : configured {TRADE_LEVERAGE}x")
                print("Quantity           : EXCHANGE MINIMUM")
                print("Orders             : ENABLED" if ORDERS_ENABLED else "Orders             : DISABLED")
                print("=" * 68)

                opened = open_position(
                    symbol,
                    side,
                    trigger,
                    stop_loss,
                    current_start,
                )

                if opened:
                    with state_lock:
                        symbol_states[symbol]["signal_triggered"] = True

        with state_lock:
            symbol_states[symbol]["previous_live_price"] = price

    except Exception as e:

        print(f"[Price Handler Error] ({symbol}) {e}")


_gap_notes = set()


def _setup_candles_ok(current_start, closed3):
    """True when the 3 reference candles + the current one are back-to-back."""
    try:
        if len(closed3) != 3:
            return False
        starts = [int(c["start_time"]) for c in closed3]
        cur = int(current_start)
        unit = 1000 if cur > 10 ** 11 else 1
        step = int(float(INTERVAL)) * 60 * unit
        seq = starts + [cur]
        return all(abs((seq[i + 1] - seq[i]) - step) <= step * 0.1 for i in range(3))
    except Exception:
        return True


def _note_gap_skip(symbol, current_start):
    key = (symbol, current_start)
    if key in _gap_notes:
        return
    if len(_gap_notes) > 2000:
        _gap_notes.clear()
    _gap_notes.add(key)
    print(f"[GAP GUARD] {symbol}: setup ignored - reference candles are not contiguous.")


def detect_cross_candidate(symbol, data):
    """Read-only body-cross snapshot used to rank simultaneous entries."""
    try:
        price = data.get("price")
        if price is None:
            return None

        with state_lock:
            state = symbol_states.get(symbol)
            if not state or symbol not in watchlist or state["in_position"]:
                return None
            if state["previous_closed_candle"] is None or state["current_candle"] is None:
                return None
            if state["signal_triggered"]:
                return None

            prev = dict(state["previous_closed_candle"])
            current = dict(state["current_candle"])
            current_start = current["start_time"]
            if current_start == prev.get("start_time"):
                return None

            old_price = state.get("previous_live_price")
            closed3 = _last_closed_candles(state, 3)

        reference_open = float(prev["open"])
        current_open = float(current["open"])
        if old_price is None:
            return None
        if len(closed3) != 3:
            return None
        colours = []
        for candle in closed3:
            co, cc = float(candle["open"]), float(candle["close"])
            colours.append("GREEN" if cc > co else ("RED" if cc < co else "DOJI"))

        if REQUIRE_CONTIGUOUS_CANDLES and not _setup_candles_ok(current_start, closed3):
            return None

        if (
            current_open <= reference_open
            and float(old_price) <= reference_open < float(price)
            and all(c == "RED" for c in colours)
        ):
            side = "BUY"
        elif (
            current_open >= reference_open
            and float(old_price) >= reference_open > float(price)
            and all(c == "GREEN" for c in colours)
        ):
            side = "SELL"
        else:
            return None

        allowed, score, reason = hft_entry_gate(symbol, side)
        if not allowed:
            return None

        trigger = float(price)
        stop_loss = _roi_price_for_target(
            trigger, side, -PRIMARY_STOP_LOSS_ROI_PCT,
            leverage=normal_entry_leverage(symbol),
        )
        if stop_loss is None:
            return None

        return {
            "symbol": symbol,
            "side": side,
            "score": score,
            "reason": reason,
            "candle": current_start,
            "entry": trigger,
            "sl": stop_loss,
        }
    except Exception:
        return None


def ticker_poll_loop():
    """Poll live prices and give the strongest simultaneous crossover first priority."""
    global entry_priority_symbol, entry_priority_score

    while True:
        all_data = fetch_all_pairs_ticker()

        if all_data:
            with state_lock:
                tracked = list(subscribed_symbols)
                for _s in tracked:
                    _d = all_data.get(_s)
                    if _d is not None:
                        ticker_extra[_s] = {
                            "funding": _d.get("funding"),
                            "next_funding_ms": _d.get("next_funding"),
                            "oi_value": _d.get("oi_value"),
                            "pct24h": _d.get("pct24h"),
                            "quote_vol24": _d.get("quote_vol24"),
                        }

            # Snapshot all possible crossovers BEFORE mutating previous_live_price.
            candidates = []
            blocked, _, _, _ = funding_status()
            if not blocked:
                for symbol in tracked:
                    data = all_data.get(symbol)
                    if data is None:
                        continue
                    candidate = detect_cross_candidate(symbol, data)
                    if candidate:
                        candidates.append(candidate)

            # Rank only candidates that are actually executable under
            # the same entry-feasibility rules used by open_position().
            # A high scanner score must never block an executable
            # lower-score candidate merely because its own entry later
            # fails min_qty, capital, leverage, or SL-risk checks.
            executable_candidates = []

            for candidate in candidates:
                symbol = candidate["symbol"]
                side = candidate["side"]
                entry = candidate["entry"]
                sl = candidate["sl"]

                feasible = False

                try:
                    if not ORDERS_ENABLED:
                        feasible = False
                    elif LIVE_MODE and (
                        account_balance is None or account_balance <= 0
                    ):
                        feasible = False
                    elif LIVE_MODE and RECOVERY_LOCK:
                        feasible = False
                    elif LOW_CAPITAL_MODE:
                        selected_lev, selected_qty, selected_margin = (
                            select_low_capital_entry(
                                symbol,
                                entry,
                                sl,
                                side,
                            )
                        )
                        feasible = (
                            selected_lev is not None
                            and selected_qty is not None
                            and selected_margin is not None
                        )
                    else:
                        selected_lev = normal_entry_leverage(symbol)

                        risk_ok = sl_risk_allowed(
                            entry,
                            sl,
                            leverage=selected_lev,
                            symbol=symbol,
                            side=side,
                        )

                        # can_afford_new_position() ultimately reads
                        # total_deployed_margin(), whose contract requires
                        # state_lock.
                        with state_lock:
                            affordable = can_afford_new_position(
                                symbol,
                                entry,
                                leverage=selected_lev,
                            )

                        feasible = risk_ok and affordable
                except Exception as e:
                    print(
                        f"[ENTRY PRIORITY] {symbol}: feasibility check "
                        f"failed -> {e}"
                    )
                    feasible = False

                if feasible and AI_MODE:
                    try:
                        _fx = ai_build_features(symbol, side, entry)
                        if _fx is not None:
                            _veto, _thr, _why = ai_should_veto(ai_predict(_fx))
                            if _veto:
                                feasible = False
                    except Exception:
                        pass

                if feasible:
                    executable_candidates.append(candidate)
                else:
                    print(
                        f"[ENTRY PRIORITY] {symbol} {side} skipped: "
                        f"not currently executable"
                    )

            executable_candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            winner = (
                executable_candidates[0]
                if executable_candidates
                else None
            )

            entry_priority_symbol = (
                winner["symbol"] if winner else None
            )
            entry_priority_score = (
                winner["score"] if winner else None
            )

            for symbol in tracked:
                data = all_data.get(symbol)
                if data is None:
                    continue
                handle_new_price(
                    symbol,
                    data["price"],
                    bid=data["bid"],
                    ask=data["ask"],
                    mark=data["mark"],
                    trade_time=data["timestamp"],
                )

            entry_priority_symbol = None
            entry_priority_score = None

        time.sleep(TICKER_POLL_SECONDS)


# ============================================================
# KLINE WEBSOCKET

@sio.on("FETCH_CANDLESTICK_CS_PRO", namespace=NAMESPACE)
def on_kline(data):

    try:

        if not isinstance(data, dict):
            return

        symbol = data.get("s")
        start_time = to_int(data.get("t"))
        open_price = to_float(data.get("o"))
        high_price = to_float(data.get("h"))
        low_price = to_float(data.get("l"))
        close_price = to_float(data.get("c"))
        volume = to_float(data.get("v"))

        candle_closed = bool(data.get("x", False))

        if (
            symbol is None
            or start_time is None
            or open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
        ):
            return

        new_candle = {
            "start_time": start_time,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }

        with state_lock:

            if symbol not in symbol_states:
                # Not (or no longer) something we track - ignore.
                return

            state = symbol_states[symbol]

            old_candle = (
                dict(state["current_candle"])
                if state["current_candle"] is not None
                else None
            )

            # ------------------------------------------------
            # NEW 5-MINUTE CANDLE
            # ------------------------------------------------

            if (
                old_candle is not None
                and old_candle["start_time"] != start_time
            ):

                state["previous_closed_candle"] = dict(old_candle)
                _remember_closed_candle(state, old_candle)

                # New candle gets a fresh signal window. This does
                # NOT touch an already-open position - it stays
                # open across candle boundaries (no fixed TP).
                state["signal"] = "WAIT"
                state["entry_price"] = None
                state["stop_loss"] = None
                state["signal_candle_start"] = None
                state["signal_triggered"] = False

                if state["live_price"] is not None:
                    state["previous_live_price"] = state["live_price"]

            # ------------------------------------------------
            # EXPLICIT CLOSED CANDLE
            # ------------------------------------------------

            if candle_closed:
                state["previous_closed_candle"] = dict(new_candle)
                _remember_closed_candle(state, new_candle)

            state["current_candle"] = new_candle

    except Exception as e:

        print(f"[Kline Handler Error] {e}")


# ============================================================
# TICKER WEBSOCKET
# ============================================================

@sio.on("FETCH_TICKER_INFO_CS_PRO", namespace=NAMESPACE)
def on_ticker(data):
    """
    Live-price push from the official Futures Ticker WebSocket.
    Feeds the same handle_new_price() pipeline that ticker_poll_loop()
    (REST) already feeds, so a WS push and a REST poll for the same
    symbol are handled identically - handle_new_price() only acts on
    price CHANGES relative to the last stored price, so getting an
    update from two sources is safe, just more responsive.

    NOTE: CoinSwitch's public docs for this event show the subscribe
    call but not a field-by-field schema for the push payload, so
    this parses defensively - it tries several plausible key names
    (matching the same defensive style already used elsewhere in this
    file, e.g. extract_actual_fee()) rather than assuming one exact
    shape. If price still doesn't move on this feed, print(data) once
    to see the real keys and adjust the *_KEYS tuples below.
    """

    try:

        if not isinstance(data, dict):
            return

        symbol = data.get("s") or data.get("m") or data.get("symbol")
        if not symbol:
            return

        symbol = str(symbol).upper().replace("/", "")

        PRICE_KEYS = ("last_price", "price", "c", "p", "ltp")
        BID_KEYS = ("best_bid_price", "bid", "b")
        ASK_KEYS = ("best_ask_price", "ask", "a")
        MARK_KEYS = ("mark_price", "mark", "i")
        TIME_KEYS = ("timestamp", "t", "E", "trade_time")

        def first_float(keys):
            for k in keys:
                v = to_float(data.get(k))
                if v is not None:
                    return v
            return None

        def first_int(keys):
            for k in keys:
                v = to_int(data.get(k))
                if v is not None:
                    return v
            return None

        price = first_float(PRICE_KEYS)
        if price is None:
            return

        handle_new_price(
            symbol,
            price,
            bid=first_float(BID_KEYS),
            ask=first_float(ASK_KEYS),
            mark=first_float(MARK_KEYS),
            trade_time=first_int(TIME_KEYS),
        )

    except Exception as e:

        print(f"[Ticker Handler Error] {e}")


# ============================================================
# WEBSOCKET CONNECTION
# ============================================================

@sio.event(namespace=NAMESPACE)
def connect():

    global ws_connected

    ws_connected = True

    print()
    print("[WebSocket] CONNECTED")
    with state_lock:
        _already = len(subscribed_symbols)
    if _already:
        print(f"[WebSocket] Reconnected - re-subscribing {_already} symbols...")
        threading.Thread(
            target=resubscribe_all_streams, daemon=True, name="ws-resubscribe",
        ).start()
    else:
        print(
            "[WebSocket] Waiting for watchlist refresh to subscribe "
            "to candle streams..."
        )


@sio.event(namespace=NAMESPACE)
def disconnect():

    global ws_connected

    ws_connected = False

    print()
    print("[WebSocket] DISCONNECTED")


@sio.event(namespace=NAMESPACE)
def connect_error(data):

    global ws_connected

    ws_connected = False

    print()
    print(f"[WebSocket] CONNECTION ERROR: {data}")


# ============================================================
# DISPLAY
# ============================================================
#
# Pure presentation layer for the terminal dashboard. Nothing here
# reads/writes trading state or changes any strategy/logic — it only
# formats the same values that were already computed in
# display_market() into a colored, boxed, mobile-friendly layout.

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    MAGENTA = "\033[95m"

    # Dashboard theme
    DARK_GREEN = "\033[32m"       # Positive P&L
    DARK_BLUE = "\033[34m"        # Funding / dashboard accent
    AMBER = "\033[38;5;214m"      # DOJI / neutral signal


def _vlen(text):
    """Visible length of a string, ignoring ANSI color codes."""
    return len(_ANSI_RE.sub("", text))


def _pad(text, width):
    pad = width - _vlen(text)
    return text + (" " * pad if pad > 0 else "")


def _color(text, color):
    return f"{color}{text}{Ansi.RESET}"


def _pnl_color(value):
    if value > 0:
        return Ansi.DARK_GREEN
    if value < 0:
        return Ansi.RED
    return Ansi.GRAY


def _dashboard_width():
    try:
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        cols = 80
    return max(20, cols)
# ---- outer box (white, double border) ----

def _outer_top(w):
    return _color("╔" + "═" * (w - 2) + "╗", Ansi.WHITE)


def _outer_bottom(w):
    return _color("╚" + "═" * (w - 2) + "╝", Ansi.WHITE)


def _outer_div(w):
    return _color("╟" + "─" * (w - 2) + "╢", Ansi.WHITE)


def _outer_line(w, text=""):
    content_w = w - 4
    return (
        _color("║", Ansi.WHITE) + " " + _pad(text, content_w) + " " + _color("║", Ansi.WHITE)
    )


def _outer_title(w, text, color=Ansi.BOLD + Ansi.YELLOW):
    content_w = w - 4
    stripped = _vlen(text)
    total_pad = max(0, content_w - stripped)
    left = total_pad // 2
    right = total_pad - left
    centered = (" " * left) + _color(text, color) + (" " * right)
    return _outer_line(w, centered)


def _outer_blank(w):
    return _outer_line(w, "")


# ---- inner box (cyan, single border) nested inside the outer box ----

def _inner_top(w, label=""):
    content_w = w - 4
    if label:
        lbl = f" {_color(label, Ansi.BOLD + Ansi.CYAN)} "
        dash_total = max(0, content_w - 2 - _vlen(lbl))
        left = 2
        right = max(0, dash_total - left)
        line = (
            _color("┌" + "─" * left, Ansi.WHITE)
            + lbl
            + _color("─" * right + "┐", Ansi.WHITE)
        )
    else:
        line = _color("┌" + "─" * (content_w - 2) + "┐", Ansi.WHITE)
    return _outer_line(w, line)


def _inner_bottom(w):
    content_w = w - 4
    return _outer_line(
        w,
        _color("└" + "─" * (content_w - 2) + "┘", Ansi.WHITE)
    )


def _wrap_ansi(text, width):
    """Word-wrap text that may contain ANSI colour codes, keeping colours
    correct across the line breaks. Returns a list of lines."""
    width = max(1, int(width))
    if not text:
        return [""]

    toks = []
    for piece in re.split(r"(\033\[[0-9;]*m)", text):
        if not piece:
            continue
        if piece.startswith("\033["):
            toks.append((True, piece))
        else:
            toks.extend((False, ch) for ch in piece)

    words = [[]]
    for t in toks:
        if (not t[0]) and t[1] == " ":
            words.append([])
        else:
            words[-1].append(t)

    def render(word, act):
        out, vis = "", 0
        for is_code, val in word:
            if is_code:
                act = "" if val == Ansi.RESET else act + val
                out += val
            else:
                out += val
                vis += 1
        return out, vis, act

    lines = []
    active = ""
    cur = ""
    cur_vis = 0
    cur_has = False

    for word in words:
        chunks = [word]
        if sum(1 for t in word if not t[0]) > width:
            chunks, piece_toks, n = [], [], 0
            for t in word:
                piece_toks.append(t)
                if not t[0]:
                    n += 1
                    if n == width:
                        chunks.append(piece_toks)
                        piece_toks, n = [], 0
            if piece_toks:
                chunks.append(piece_toks)

        for chunk in chunks:
            s_chunk, vis, new_active = render(chunk, active)
            need = vis + (1 if cur_has else 0)

            if cur_has and cur_vis + need > width:
                lines.append(cur + (Ansi.RESET if active else ""))
                cur, cur_vis, cur_has = active, 0, False
                need = vis

            cur += (" " if cur_has else "") + s_chunk
            cur_vis += need
            cur_has = True
            active = new_active

    lines.append(cur + (Ansi.RESET if active else ""))
    return lines


def _inner_line(w, text=""):
    """One inner-box row. Text wider than the box is word-wrapped onto
    extra rows, so nothing can spill past the right border on a phone."""
    content_w = w - 4
    inner_w = content_w - 4

    rows = _wrap_ansi(text, inner_w) if _vlen(text) > inner_w else [text]

    out = []
    for row in rows:
        body = (
            _color("│", Ansi.WHITE) + " " + _pad(row, inner_w) + " "
            + _color("│", Ansi.WHITE)
        )
        out.append(_outer_line(w, body))
    return "\n".join(out)


def _inner_wrapped(w, label, symbols, color, empty_text="none", bold=True):
    """Yield inner-box lines for a label + a wrapped, comma-joined symbol list."""
    content_w = w - 4
    inner_w = content_w - 4
    count = len(symbols)
    header_style = (Ansi.BOLD + color) if bold else color
    header = _color(f"{label} ({count}):", header_style)
    lines = [_inner_line(w, header)]
    if not symbols:
        lines.append(_inner_line(w, _color(f"  {empty_text}", Ansi.GRAY)))
        return lines
    joined = ", ".join(symbols)
    for wrapped in textwrap.wrap(joined, width=max(10, inner_w - 2)) or [""]:
        lines.append(_inner_line(w, "  " + wrapped))
    return lines


def _split_line(w, left, right):
    """Outer-box line with `left` flush left and `right` flush right on the
    same row, fully visible (no truncation) with the gap between filled in."""
    content_w = w - 4
    lv, rv = _vlen(left), _vlen(right)
    gap = max(1, content_w - lv - rv)
    return _outer_line(w, left + (" " * gap) + right)


def _pair_row(w, seg1, seg2, connector=" \u2502 "):
    """Two small connected 'boxes' (seg1, seg2) on one inner-box row when
    they fit; falls back to one segment per row if the combo is too wide."""
    content_w = w - 4
    inner_w = content_w - 4
    combined = seg1 + connector + seg2
    if _vlen(combined) <= inner_w:
        return [_inner_line(w, combined)]
    return [_inner_line(w, seg1), _inner_line(w, seg2)]


def display_market():
    with state_lock:
        connected = ws_connected
        watch_count = len(watchlist)
        tracked_count = len(subscribed_symbols)
        refresh_ts = last_watchlist_refresh
        open_positions = []

        for symbol, state in symbol_states.items():
            if state["in_position"]:
                # Display price can remain the live ticker price.
                # PnL itself MUST use the exchange mark price.
                display_price = state.get("live_price")
                pnl_price = state.get("mark_price") or display_price

                entry = state["position_entry"]
                side = state["position_side"]

                # Entry leverage is only used when opening a new position.
                # Existing-position PnL uses actual exchange leverage/margin.
                actual_leverage = state.get("actual_leverage")
                actual_margin = state.get("position_margin")

                if actual_leverage is None:
                    actual_leverage = 0.0
                else:
                    actual_leverage = float(actual_leverage)

                if actual_margin is None:
                    actual_margin = (
                        state["position_size"] / actual_leverage
                        if state.get("position_size") and actual_leverage > 0
                        else 0.0
                    )

                if LIVE_MODE:
                    # LIVE dashboard P&L must come directly from
                    # CoinSwitch exchange-reported unrealised P&L.
                    unreal = to_float(
                        state.get("exchange_unrealised_pnl")
                    )
                    if unreal is None:
                        unreal = 0.0
                else:
                    unreal = (
                        calculate_leveraged_pnl_pct(
                            entry,
                            pnl_price,
                            side,
                            leverage=actual_leverage,
                            margin=actual_margin,
                            position_size=state.get("position_size"),
                        )
                        if pnl_price is not None and entry
                        else 0.0
                    )

                open_positions.append({
                    "symbol": symbol, "side": side, "entry": entry, "price": display_price,
                    "sl": state.get("active_sl") or state.get("position_sl"),
                    "sl_type": state.get("protection_type") or (
                        "TRAILING" if state.get("trailing_active") else "INITIAL"
                    ),
                    "peak": state.get("trailing_peak"),
                    "size": state["position_size"],
                    "qty": state["position_qty"],
                    "unreal": unreal,
                    "exchange_unrealised_pnl": state.get(
                        "exchange_unrealised_pnl"
                    ),
                    "margin": actual_margin,
                    "leverage": actual_leverage,
                    "scanner_score": state.get("scanner_score"),
                    "scanner_quality": state.get("scanner_quality"),
                })

        # Mode-specific trade source for dashboard P&L/history.
        # PAPER ON -> paper data only
        # LIVE ON  -> live data only
        # Both OFF -> no P&L/history data
        if PAPER_MODE:
            pnl_history = paper_trade_log
        elif LIVE_MODE:
            pnl_history = live_trade_log
        else:
            pnl_history = []

        # Only active-mode trades created in this terminal session are displayed.
        closed_trades = [
            t for t in pnl_history
            if float(t.get("closed_at") or 0) >= SESSION_STARTED_AT
        ]

        # Current-session metrics only. The session starts when this bot
        # process starts; older persisted trades are not included.
        session_gross = sum(
            float(t.get("pnl_usdt") or 0.0)
            for t in closed_trades
        )
        session_fees = sum(
            float(t.get("fee") or 0.0)
            for t in closed_trades
        )
        session_pnl = session_gross - session_fees

        historical_count = len(pnl_history)

        missing_price, missing_candles = [], []
        red_symbols, green_symbols, doji_symbols = [], [], []
        for symbol in watchlist:
            state = symbol_states.get(symbol)
            if state is None:
                missing_price.append(symbol); missing_candles.append(symbol); continue
            if state["live_price"] is None:
                missing_price.append(symbol)
            prev = state["previous_closed_candle"]
            if prev is None:
                missing_candles.append(symbol); continue
            if state["in_position"]:
                continue
            if prev["close"] < prev["open"]: red_symbols.append(symbol)
            elif prev["close"] > prev["open"]: green_symbols.append(symbol)
            else: doji_symbols.append(symbol)

    deployed = sum(float(p.get("size") or 0.0) for p in open_positions)
    deployed_margin = sum(float(p.get("margin") or 0.0) for p in open_positions)
    total_capital = effective_total_capital()
    available_capital = effective_available_capital()
    max_margin = max_deployable_margin()
    used_pct = (deployed_margin / total_capital * 100.0) if total_capital > 0 else 0.0
    blocked, funding_time, next_funding, _ = funding_status()
    # Account / execution status for dashboard
    orders_label = "ON" if ORDERS_ENABLED else "OFF"
    paper_label = "ON" if PAPER_MODE else "OFF"
    live_label = "ON" if LIVE_MODE else "OFF"

    paper_balance = paper_capital if PAPER_MODE else 0.0
    live_balance = account_balance if (LIVE_MODE and account_balance is not None) else 0.0

    # ========================================================
    # P&L SOURCE OF TRUTH
    #
    # LIVE:
    #   CoinSwitch exchange data ONLY.
    #   Never calculate LIVE P&L from local entry/exit/price.
    #
    # PAPER:
    #   Existing local paper-trade accounting remains unchanged.
    # ========================================================

    if LIVE_MODE:
        # ----------------------------------------------------
        # LIVE UNREALIZED P&L
        # ----------------------------------------------------
        # Each live position is populated from the exchange
        # position snapshot. The exchange-reported unrealised
        # P&L is the ONLY LIVE USDT P&L value accepted here.
        unrealized_pnl_usdt = 0.0
        exchange_unrealized_count = 0

        for p in open_positions:
            state = symbol_states.get(p["symbol"], {})
            exchange_pnl = state.get("exchange_unrealised_pnl")

            try:
                if exchange_pnl is not None:
                    unrealized_pnl_usdt += float(exchange_pnl)
                    exchange_unrealized_count += 1
            except (TypeError, ValueError):
                pass

        # Never fall back to local price/entry math in LIVE mode.
        if open_positions and exchange_unrealized_count == len(open_positions):
            unrealized_pnl_pct = (
                unrealized_pnl_usdt / deployed_margin * 100.0
                if deployed_margin > 0
                else 0.0
            )
        else:
            # Exchange P&L is incomplete/unavailable.
            # Do NOT invent a local estimate.
            unrealized_pnl_usdt = 0.0
            unrealized_pnl_pct = 0.0

        # ----------------------------------------------------
        # LIVE REALIZED P&L
        # ----------------------------------------------------
        # Read CoinSwitch transaction history. Do not use the
        # locally calculated closed-trade P&L as LIVE truth.
        # Exchange-reported realized P&L comes from a background cache
        # (never fetched here, so the dashboard thread does no REST I/O).
        with state_lock:
            _cached_realized = live_realized_cache.get("usdt")

        if _cached_realized is not None:
            realized_pnl_usdt = float(_cached_realized)
        else:
            realized_pnl_usdt = 0.0

        realized_pnl_pct = (
            realized_pnl_usdt / total_capital * 100.0
            if total_capital > 0
            else 0.0
        )

        # LIVE closed-trade win count must not be inferred from
        # local P&L either.
        wins = sum(
            1 for t in closed_trades
            if float(t.get("net_pnl_usdt") or 0.0) > 0
        )

    elif PAPER_MODE:
        # ----------------------------------------------------
        # PAPER MODE: local simulated P&L is valid.
        # ----------------------------------------------------
        realized_pnl_usdt = sum(
            float(t.get("pnl_usdt") or 0.0)
            for t in closed_trades
        )

        realized_margin = sum(
            float(t.get("margin") or 0.0)
            for t in closed_trades
        )

        realized_pnl_pct = (
            realized_pnl_usdt / realized_margin * 100.0
            if realized_margin > 0
            else 0.0
        )

        wins = sum(
            1 for t in closed_trades
            if float(t.get("pnl_pct") or 0.0) > 0
        )

        unrealized_pnl_pct = sum(
            float(p.get("unreal") or 0.0)
            for p in open_positions
        )

        unrealized_pnl_usdt = 0.0

        for p in open_positions:
            pnl_price = p.get("price")

            if pnl_price is not None:
                raw_unreal = calculate_raw_pnl_pct(
                    p["entry"],
                    pnl_price,
                    p["side"]
                )
                unrealized_pnl_usdt += (
                    (p["size"] or 0.0) * raw_unreal / 100.0
                )

    else:
        realized_pnl_usdt = 0.0
        realized_pnl_pct = 0.0
        unrealized_pnl_usdt = 0.0
        unrealized_pnl_pct = 0.0
        wins = 0

    total_pnl_pct = realized_pnl_pct + unrealized_pnl_pct
    total_pnl_usdt = realized_pnl_usdt + unrealized_pnl_usdt

    w = _dashboard_width()

    # Shared dashboard dimensions. w MUST be initialized first.
    content_w = w - 4
    inner_w = content_w - 4
    L = []

    # ---- header ----
    L.append(_outer_top(w))
    L.append(_outer_title(w, "HFT PRIORITY + 5M BODY OPEN CROSS", Ansi.BOLD + Ansi.WHITE))
    L.append(_outer_title(w, f"DEFAULT {TRADE_LEVERAGE}x | ACTUAL LEVERAGE ACCOUNTING", Ansi.BOLD + Ansi.WHITE))
    L.append(_outer_div(w))

    ws_color = Ansi.GREEN if connected else Ansi.RED
    ws_text = "CONNECTED" if connected else "DISCONNECTED"
    if LIVE_MODE:
        orders_tag = _color("LIVE", Ansi.BOLD + Ansi.GREEN)
    elif PAPER_MODE:
        orders_tag = _color("PAPER", Ansi.BOLD + Ansi.YELLOW)
    else:
        orders_tag = _color("OFF", Ansi.BOLD + Ansi.GRAY)
    L.append(_split_line(w, f"WebSocket : {_color(ws_text, Ansi.BOLD + ws_color)}", orders_tag))
    L.append(_outer_line(w, f"Time (UTC): {now_string()}"))
    L.append(_outer_line(w, f"Watchlist : {watch_count} active | {tracked_count} tracked"))
    if refresh_ts:
        next_refresh_in = max(0, int(HFT_REFRESH_SECONDS - (time.time() - refresh_ts)))
        L.append(_outer_line(w, f"Next refresh in ~{next_refresh_in} s"))
    else:
        L.append(_outer_line(w, "Watchlist : not yet refreshed"))

    # ---- scan results (nested nested box = "double layer") ----
    L.append(_outer_div(w))
    data_ok = watch_count - len(set(missing_price) | set(missing_candles))
    L.append(_outer_line(w, _color(f"Scan status: {data_ok}/{watch_count} symbols ready", Ansi.BOLD + Ansi.WHITE)))
    L.append(_inner_top(w, "SCAN RESULTS"))
    L.append(_inner_line(w, "Entry: CURRENT BODY crosses PREVIOUS OPEN"))
    if missing_price:
        L.extend(_inner_wrapped(w, "NO LIVE PRICE", missing_price, Ansi.YELLOW))
    if missing_candles:
        L.extend(_inner_wrapped(w, "NO CANDLE DATA", missing_candles, Ansi.YELLOW))
    L.extend(_inner_wrapped(w, "PREV RED  - valid reference", red_symbols, Ansi.RED, bold=False))
    L.extend(_inner_wrapped(w, "PREV GREEN- valid reference", green_symbols, Ansi.GREEN, bold=False))
    if doji_symbols:
        L.extend(_inner_wrapped(w, "PREV DOJI - valid reference", doji_symbols, Ansi.AMBER, bold=False))
    L.append(_inner_bottom(w))

    # ---- closed trades (numbered, same compact connected-box format) ----
    L.append(_outer_div(w))
    L.append(_inner_top(w, f"CLOSED: {len(closed_trades)} sess / {historical_count} hist"))
    if closed_trades:
        for i, t in enumerate(closed_trades, start=1):
            side_color = Ansi.GREEN if t['side'] == "BUY" else Ansi.RED
            pnl_c = _pnl_color(float(t.get("pnl_pct") or 0.0))
            trade_pnl_str = f"{t['pnl_pct']:.2f}% ({t['pnl_usdt']:+.2f})"
            net = float(t.get('net_pnl_usdt') or 0.0)
            fee = float(t.get('fee') or 0.0)
            L.append(_inner_line(w, _color(f"#{i} [{t['side']}] {t['symbol']}", Ansi.BOLD + side_color)))
            L.append(_inner_line(
                w,
                f"Entry {t['entry']:.4f} | Exit {t['exit']:.4f} | "
                f"PnL {_color(trade_pnl_str, pnl_c)} | Fees {fee:.2f} | "
                f"Net {_color(f'{net:+.2f}', _pnl_color(net))}"
            ))
            for reason_line in textwrap.wrap(f"reason: {t['reason']}", width=max(10, (w - 4) - 4 - 2)) or []:
                L.append(_inner_line(w, "  " + reason_line))
    else:
        L.append(_inner_line(w, _color("none", Ansi.GRAY)))
    L.append(_inner_bottom(w))

    # ---- open positions (numbered, compact connected-box rows) ----
    L.append(_outer_div(w))
    L.append(_inner_top(w, f"OPEN POSITIONS ({len(open_positions)})"))
    if open_positions:
        for i, p in enumerate(open_positions, start=1):
            price_str = f"{p['price']:.4f}" if p.get('price') is not None else "N/A"
            entry_str = f"{p['entry']:.4f}" if p.get('entry') is not None else "N/A"
            sl_str = f"{p['sl']:.4f}" if p.get('sl') is not None else "N/A"
            sl_type = str(p.get('sl_type') or "INITIAL").upper()
            peak_str = f"{p['peak']:.4f}" if p.get('peak') is not None else "N/A"
            qty_str = f"{p['qty']}" if p.get('qty') is not None else "N/A"
            side_color = Ansi.GREEN if p['side'] == "BUY" else Ansi.RED
            margin_p = float(p.get("margin") or 0.0)
            cap_used = (margin_p / total_capital * 100.0) if total_capital else 0.0
            unreal_str = f"{float(p.get('unreal') or 0.0):.2f}%"
            L.append(_inner_line(w, _color(f"#{i} [{p['side']}] {p['symbol']}", Ansi.BOLD + side_color)))
            # Wrap position details to the actual inner-box width so no
            # position data can spill past the right border on narrow
            # Termux/mobile terminals. Keep the fields in the same order.
            position_details = (
                f"Qty {qty_str} | Entry {entry_str} | Price {price_str} | "
                f"SL {sl_str} [{sl_type}] | Peak {peak_str} | "
                f"Margin {margin_p:.2f} ({cap_used:.2f}%) | "
                f"PnL {_color(unreal_str, _pnl_color(p['unreal']))}"
            )
            for detail_line in textwrap.wrap(
                position_details,
                width=max(10, inner_w),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]:
                L.append(_inner_line(w, detail_line))

    else:
        L.append(_inner_line(w, _color("none", Ansi.GRAY)))
    L.append(_inner_bottom(w))

    # ---- shared dashboard width calculations ----
    # Must be defined before OPEN/CLOSED/STATUS/CAPITAL/P&L blocks.
    # ---- STATUS ----
    L.append(_outer_div(w))
    L.append(_inner_top(w, "STATUS"))

    def _onoff(label, on, color):
        txt = _color("ON", Ansi.BOLD + color) if on else _color("OFF", Ansi.GRAY)
        return f"{label} {txt}"

    _status_items = (
        ("ORDERS", ORDERS_ENABLED, Ansi.GREEN),
        ("PAPER", PAPER_MODE, Ansi.YELLOW),
        ("LIVE", LIVE_MODE, Ansi.GREEN),
    )
    _short_names = {"ORDERS": "ORD", "PAPER": "PAP", "LIVE": "LIV"}
    ai_right = _onoff("AI", AI_MODE, Ansi.MAGENTA)

    # ORDERS / PAPER / LIVE on the left, AI flush RIGHT on the same row.
    # Labels shrink automatically if the phone screen is narrow.
    _row_done = False
    left = ""
    for _sep, _use_short in (("  ", False), (" ", False), (" ", True)):
        left = _sep.join(
            _onoff(_short_names[nm] if _use_short else nm, on, col)
            for nm, on, col in _status_items
        )
        if _vlen(left) + _vlen(ai_right) + 1 <= inner_w:
            gap = inner_w - _vlen(left) - _vlen(ai_right)
            L.append(_inner_line(w, left + (" " * gap) + ai_right))
            _row_done = True
            break

    if not _row_done:
        L.append(_inner_line(w, left))
        L.append(_inner_line(
            w, (" " * max(0, inner_w - _vlen(ai_right))) + ai_right
        ))

    for _ai_text in ai_status_lines():
        for _ln in textwrap.wrap(_ai_text, width=max(10, inner_w)):
            L.append(_inner_line(w, _ln))

    for _x_text in (api_usage_lines() + [risk_status_line()]):
        for _ln in textwrap.wrap(_x_text, width=max(10, inner_w)):
            L.append(_inner_line(w, _ln))

    L.append(_inner_bottom(w))

    # ---- CAPITAL ----
    L.append(_inner_top(w, "CAPITAL"))

    L.append(_inner_line(
        w,
        f"Capital   : {total_capital:.2f} USDT"
    ))
    L.append(_inner_line(
        w,
        f"Available : {available_capital:.2f} USDT"
    ))
    # Margin row + CAP on the SAME row.
    margin_left = f"Margin    : {deployed_margin:.2f} / {max_margin:.2f} USDT"

    if LOW_CAPITAL_MODE:
        cap_right = f"C-MODE | {used_pct:.2f}% used"
    else:
        cap_right = f"{MAX_ALLOCATION_PCT:.0f}% CAP | {used_pct:.2f}% used"

    margin_text = margin_left + "  " + cap_right

    L.append(_inner_line(w, margin_text))

    # Live C / M mode status.
    if LOW_CAPITAL_MODE:
        c_mode_text = _color(
            "C-MODE    : ON  | MIN QTY | 10->20->40x",
            Ansi.DARK_BLUE
        )
    else:
        c_mode_text = "C-MODE    : OFF | NORMAL 50% CAP"

    if MAX_LEVERAGE_MODE:
        m_mode_text = _color(
            "M-MODE    : ON  | COIN MAX LEVERAGE",
            Ansi.DARK_BLUE
        )
    else:
        m_mode_text = "M-MODE    : OFF | FIXED 10x"

    L.append(_inner_line(w, c_mode_text))
    L.append(_inner_line(w, m_mode_text))

    L.append(_inner_line(
        w,
        f"Notional  : {deployed:.2f} USDT @ actual leverage"
    ))
    L.append(_inner_line(
        w,
        f"Positions : {len(open_positions)} / UNLIMITED"
    ))
    L.append(_inner_line(
        w,
        f"Max SL    : {MAX_SL_RISK_PCT:.1f}% of margin"
    ))

    fund_color = Ansi.RED if blocked else Ansi.DARK_BLUE
    fund_text = "BLOCKED" if blocked else "ENTRY ACTIVE"

    L.append(_inner_line(
        w,
        f"Funding   : {_color(fund_text, Ansi.BOLD + fund_color)}"
    ))
    L.append(_inner_line(
        w,
        f"Next      : {(next_funding or funding_time).strftime('%H:%M UTC')} "
        f"(±{FUNDING_BLACKOUT_MINUTES}m)"
    ))

    L.append(_inner_bottom(w))

    # ---- P&L ----
    L.append(_inner_top(w, "P&L"))

    realized_value = _color(
        f"{realized_pnl_pct:+.2f}% ({realized_pnl_usdt:+.2f})",
        _pnl_color(realized_pnl_pct)
    )

    unrealized_value = _color(
        f"{unrealized_pnl_pct:+.2f}% ({unrealized_pnl_usdt:+.2f})",
        _pnl_color(unrealized_pnl_pct)
    )

    total_value = _color(
        f"{total_pnl_pct:+.2f}% ({total_pnl_usdt:+.2f})",
        Ansi.BOLD + _pnl_color(total_pnl_pct)
    )

    # Session PnL is NET of session fees, and its percentage is measured
    # against the configured session capital rather than position margin.
    session_pnl_capital_pct = (
        session_pnl / total_capital * 100.0
        if total_capital > 0
        else 0.0
    )

    session_pnl_value = _color(
        f"{session_pnl:+.2f} USDT ({session_pnl_capital_pct:+.2f}%)",
        _pnl_color(session_pnl_capital_pct)
    )
    session_fees_value = _color(
        f"{session_fees:.2f} USDT",
        Ansi.RESET
    )

    session_pnl_text = _color(
        f"Session PnL: {session_pnl_value}",
        Ansi.RESET
    )
    session_fees_text = _color(
        f"Fees Sess.: {session_fees_value}",
        Ansi.RESET
    )

    # Two-column rows INSIDE the inner P&L box.
    # Left side stays aligned; right side is flush to the inner-box edge.

    left_realized = f"Realized   : {realized_value}"
    left_unrealized = f"Unrealized : {unrealized_value}"

    right_session_pnl = session_pnl_text
    right_session_fees = session_fees_text

    def _pnl_inner_row(left, right):
        # Side by side when they fit; otherwise one under the other.
        if _vlen(left) + _vlen(right) + 1 > inner_w:
            return _inner_line(w, left) + "\n" + _inner_line(w, right)
        gap = inner_w - _vlen(left) - _vlen(right)
        return _inner_line(w, left + (" " * gap) + right)

    L.append(_pnl_inner_row(left_realized, right_session_pnl))
    L.append(_pnl_inner_row(left_unrealized, right_session_fees))

    L.append(_inner_line(
        w,
        _color(f"TOTAL      : {total_value}", Ansi.BOLD)
    ))

    if closed_trades:
        L.append(_inner_line(
            w,
            f"Win rate   : {wins}/{len(closed_trades)}"
        ))

    L.append(_inner_bottom(w))

    # ---- keyboard controls / dashboard tip ----
    _tip = (
        "TIP: P=Paper | L=Live | O=Orders | C=LowCap | M=MaxLev | "
        "A=AI | R=Recovery | E=Exit | Q=Quit"
    )
    for _ln in textwrap.wrap(_tip, width=max(10, inner_w)):
        L.append(_inner_line(w, _ln))

    L.append(_outer_bottom(w))

    # ---- fixed top-anchored dashboard redraw ----
    print("\033[2J\033[H", end="")
    print("\n".join(L))


def _keyboard_control_loop():
    """
    Non-blocking terminal keyboard controls for the dashboard.

    P = toggle paper mode
    L = toggle live mode (requires Y confirmation when enabling)
    O = toggle order execution
    M = toggle MAX LEVERAGE MODE (Y/N confirmation)
    C = toggle LOW CAPITAL MODE (Y/N confirmation)
    M = toggle MAX LEVERAGE MODE for NORMAL entries
    A = toggle AI mode (local learning model)
    R = run LIVE exchange recovery
    E = emergency/manual exit ALL open positions
    Q = stop the bot
    """
    import sys
    import select
    import termios
    import tty

    global PAPER_MODE, LIVE_MODE, ORDERS_ENABLED, LOW_CAPITAL_MODE, MAX_LEVERAGE_MODE, AI_MODE

    fd = sys.stdin.fileno()

    try:
        old_settings = termios.tcgetattr(fd)
    except Exception as e:
        print(f"[Keyboard] Terminal control unavailable: {e}")
        return

    try:
        tty.setcbreak(fd)

        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)

            if not readable:
                continue

            key = sys.stdin.read(1).lower()

            # -------------------------------------------------
            # Q = STOP
            # -------------------------------------------------
            if key == "q":
                print("\n[Keyboard] Q pressed - stopping bot...")

                try:
                    sio.disconnect()
                except Exception:
                    pass

                return

            # -------------------------------------------------
            # P = PAPER TOGGLE
            # -------------------------------------------------
            if key == "p":

                if PAPER_MODE:
                    PAPER_MODE = False
                    print("\n[Keyboard] PAPER_MODE = OFF")

                else:
                    if LIVE_MODE:
                        LIVE_MODE = False
                        print("\n[Keyboard] LIVE_MODE = OFF")

                    PAPER_MODE = True
                    print("\n[Keyboard] PAPER_MODE = ON")

                continue

            # -------------------------------------------------
            # L = LIVE TOGGLE
            # -------------------------------------------------
            if key == "l":

                if LIVE_MODE:
                    LIVE_MODE = False
                    print("\n[Keyboard] LIVE_MODE = OFF")
                    continue

                print("\n" + "!" * 68)
                print("!!! LIVE MODE REQUESTED !!!")
                print("LIVE mode can use REAL CoinSwitch futures execution.")
                print("This does NOT automatically enable ORDERS.")
                print("Press Y to confirm LIVE_MODE ON.")
                print("Any other key cancels.")
                print("!" * 68)

                confirm = sys.stdin.read(1).lower()

                if confirm == "y":

                    PAPER_MODE = False
                    LIVE_MODE = True

                    print("[Keyboard] LIVE_MODE = ON")

                    try:
                        refresh_account_balance()
                    except Exception as e:
                        print(f"[Keyboard] Balance refresh failed: {e}")

                    # -------------------------------------------------
                    # Automatic LIVE exchange recovery BEFORE orders.
                    # ORDERS_ENABLED is still OFF, so this is strictly
                    # read-only discovery/synchronization.
                    # -------------------------------------------------
                    try:
                        recovered = recover_live_positions(
                            allow_orders_off=True
                        )

                        if recovered:
                            print(
                                "[Keyboard] AUTOMATIC LIVE recovery completed: "
                                "exchange positions discovered/synced; "
                                "no orders or protection actions were attempted."
                            )
                        else:
                            print(
                                "[Keyboard] AUTOMATIC LIVE recovery completed "
                                "with RECOVERY_LOCK still active."
                            )

                    except Exception as e:
                        print(f"[Keyboard] Automatic LIVE recovery failed: {e}")

                    # Start continuous read-only exchange position sync.
                    start_live_position_sync_thread()
                    if ORDERS_ENABLED:
                        start_protection_watchdog_thread()

                else:
                    print("[Keyboard] LIVE_MODE activation cancelled.")

                continue

            # -------------------------------------------------
            # O = ORDERS TOGGLE
            # -------------------------------------------------
            if key == "o":

                if ORDERS_ENABLED:
                    ORDERS_ENABLED = False
                    print("\n[Keyboard] ORDERS_ENABLED = OFF")

                else:
                    if LIVE_MODE:
                        print("\n" + "!" * 68)
                        print("!!! REAL ORDERS REQUESTED !!!")
                        print("LIVE_MODE is ON: enabling ORDERS places REAL")
                        print("MARKET orders with REAL money.")
                        print("Press Y to confirm. Any other key cancels.")
                        print("!" * 68)

                        confirm = sys.stdin.read(1).lower()

                        if confirm != "y":
                            print("[Keyboard] ORDERS activation cancelled.")
                            continue

                    ORDERS_ENABLED = True

                    if LIVE_MODE:
                        print("\n[Keyboard] ORDERS_ENABLED = ON")
                        print("[Keyboard] LIVE_MODE is ON - REAL orders are now permitted.")

                        # Reconcile + protect exchange positions BEFORE trading,
                        # and make sure the live threads are running.
                        try:
                            recover_live_positions()
                        except Exception as e:
                            print(f"[Keyboard] LIVE recovery failed: {e}")

                        start_live_position_sync_thread()
                        start_protection_watchdog_thread()

                    elif PAPER_MODE:
                        print("\n[Keyboard] ORDERS_ENABLED = ON")
                        print("[Keyboard] PAPER trading orders enabled.")

                    else:
                        print("\n[Keyboard] ORDERS_ENABLED = ON")
                        print("[Keyboard] No PAPER/LIVE mode selected; no trade execution mode is active.")

                continue

            # -------------------------------------------------
            # E = EMERGENCY EXIT ALL OPEN POSITIONS
            # -------------------------------------------------
            if key == "e":

                print("\n" + "!" * 68)
                print("!!! EMERGENCY EXIT ALL OPEN POSITIONS !!!")

                with state_lock:
                    open_positions = []

                    for symbol, state in symbol_states.items():
                        if state.get("in_position") and not state.get("position_closing"):
                            exit_price = (
                                state.get("live_price")
                                or state.get("mark_price")
                            )

                            open_positions.append(
                                (symbol, exit_price)
                            )

                if not open_positions:
                    print("[Keyboard] No open positions found.")
                    print("!" * 68)
                    continue

                print(
                    f"[Keyboard] {len(open_positions)} open position(s) "
                    "will be sent to close_position()."
                )

                for symbol, exit_price in open_positions:
                    print(
                        f"  - {symbol}: exit reference = "
                        f"{exit_price if exit_price is not None else 'N/A'}"
                    )

                print()
                print("Confirm EMERGENCY EXIT ALL?")
                print("Press Y = YES, execute all closes")
                print("Press N = NO, cancel")
                print("!" * 68)

                confirm = sys.stdin.read(1).lower()

                if confirm != "y":
                    print("[Keyboard] Emergency exit cancelled.")
                    continue

                print(
                    "\n[Keyboard] EMERGENCY EXIT CONFIRMED - "
                    "closing all open positions..."
                )

                for symbol, exit_price in open_positions:

                    if exit_price is None:
                        print(
                            f"[Keyboard] {symbol}: no live/mark price available; "
                            "close skipped."
                        )
                        continue

                    try:
                        close_position(
                            symbol,
                            exit_price,
                            "MANUAL EMERGENCY EXIT ALL"
                        )

                    except Exception as e:
                        print(
                            f"[Keyboard] Emergency close failed for "
                            f"{symbol}: {e}"
                        )

                print(
                    "[Keyboard] EMERGENCY EXIT ALL sequence completed."
                )

                continue

            # -------------------------------------------------
            # M = MAX LEVERAGE MODE
            # Requires Y confirmation before changing state.
            # C and M are mutually exclusive.
            # -------------------------------------------------
            if key == "m":

                requested = not MAX_LEVERAGE_MODE

                print("\n" + "=" * 68)
                print(
                    "[Keyboard] MAX LEVERAGE MODE -> "
                    f"{'ON' if requested else 'OFF'} requested"
                )
                print("Press Y = YES")
                print("Press N = NO")
                print("=" * 68)

                confirm = sys.stdin.read(1).lower()

                if confirm == "y":
                    MAX_LEVERAGE_MODE = requested

                    if MAX_LEVERAGE_MODE:
                        if LOW_CAPITAL_MODE:
                            LOW_CAPITAL_MODE = False
                            print(
                                "[Keyboard] C/LOW CAPITAL MODE = OFF | "
                                "M/MAX LEVERAGE MODE activated."
                            )

                        print(
                            "[Keyboard] MAX LEVERAGE MODE = ON | "
                            "NORMAL entries use each coin's MAX leverage"
                        )
                    else:
                        print(
                            "[Keyboard] MAX LEVERAGE MODE = OFF | "
                            f"NORMAL entries use configured "
                            f"{TRADE_LEVERAGE:g}x"
                        )
                else:
                    print(
                        "[Keyboard] MAX LEVERAGE MODE change cancelled."
                    )

                continue

            # -------------------------------------------------
            # C = LOW CAPITAL MODE
            # Requires Y confirmation before changing state.
            # C and M are mutually exclusive.
            # -------------------------------------------------
            if key == "c":

                requested = not LOW_CAPITAL_MODE

                print("\n" + "=" * 68)
                print(
                    "[Keyboard] LOW CAPITAL MODE -> "
                    f"{'ON' if requested else 'OFF'} requested"
                )
                print("Press Y = YES")
                print("Press N = NO")
                print("=" * 68)

                confirm = sys.stdin.read(1).lower()

                if confirm == "y":
                    LOW_CAPITAL_MODE = requested

                    if LOW_CAPITAL_MODE:
                        if MAX_LEVERAGE_MODE:
                            MAX_LEVERAGE_MODE = False
                            print(
                                "[Keyboard] M/MAX LEVERAGE MODE = OFF | "
                                "C/LOW CAPITAL MODE activated."
                            )

                        print(
                            "[Keyboard] LOW CAPITAL MODE = ON | "
                            "min qty | leverage fallback 10x -> 20x -> 40x"
                        )
                    else:
                        print(
                            "[Keyboard] LOW CAPITAL MODE = OFF | "
                            f"normal {TRADE_LEVERAGE:g}x / 50% cap"
                        )
                else:
                    print(
                        "[Keyboard] LOW CAPITAL MODE change cancelled."
                    )

                continue

            # -------------------------------------------------
            # A = AI MODE TOGGLE
            # -------------------------------------------------
            if key == "a":
                AI_MODE = not AI_MODE
                print(f"\n[Keyboard] AI_MODE = {'ON' if AI_MODE else 'OFF'}")

                if AI_MODE:
                    line1, _ = ai_status_lines()
                    print(f"[Keyboard] {line1}")
                    print(
                        "[Keyboard] AI can only SKIP entries; it never "
                        "opens trades or changes stop-losses."
                    )

                continue

            if key == "r":

                if not LIVE_MODE:
                    print(
                        "\n[Keyboard] LIVE recovery blocked: "
                        "LIVE_MODE must be ON."
                    )
                    continue

                try:
                    recovered = recover_live_positions(
                        allow_orders_off=True
                    )

                    if recovered:
                        if ORDERS_ENABLED:
                            print(
                                "[Keyboard] LIVE recovery completed: "
                                "all exchange positions protected."
                            )
                        else:
                            print(
                                "[Keyboard] READ-ONLY LIVE recovery completed: "
                                "exchange positions discovered/synced; "
                                "no orders or protection actions were attempted."
                            )
                    else:
                        print(
                            "[Keyboard] LIVE recovery completed with "
                            "RECOVERY_LOCK still active."
                        )

                except Exception as e:
                    print(f"[Keyboard] LIVE recovery failed: {e}")

                continue

    except Exception as e:
        print(f"[Keyboard Control Error] {e}")

    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

def display_loop():

    keyboard_thread = threading.Thread(
        target=_keyboard_control_loop,
        daemon=True,
        name="dashboard-keyboard",
    )
    keyboard_thread.start()

    while True:

        try:
            display_market()
        except Exception as e:
            print(f"[Display Loop Error] {e}")

        time.sleep(PRINT_INTERVAL)


def _settle_open_paper_positions_on_shutdown():
    """Stop new entries and settle every open PAPER position at a valid market price."""
    global BOT_SHUTTING_DOWN
    with state_lock:
        BOT_SHUTTING_DOWN = True

    if not PAPER_MODE:
        return

    with state_lock:
        pending = [
            (symbol, state.get("live_price") or state.get("mark_price"))
            for symbol, state in symbol_states.items()
            if state.get("in_position") and not state.get("position_closing")
        ]

    if not pending:
        return

    # Try to obtain a fresh market quote for every symbol; fall back only
    # to a previously observed valid price if the refresh is unavailable.
    refreshed = {}
    try:
        ticker_data = fetch_all_pairs_ticker() or {}
        for symbol, row in ticker_data.items():
            if isinstance(row, dict):
                refreshed[symbol] = to_float(row.get("price"))
    except Exception as exc:
        print(f"[PAPER SHUTDOWN] Latest ticker refresh failed: {exc}")

    print(f"[PAPER SHUTDOWN] Settling {len(pending)} open paper position(s)...")
    for symbol, price in pending:
        fresh_price = refreshed.get(symbol)
        if fresh_price is not None and fresh_price > 0:
            price = fresh_price
        if price is None or float(price or 0.0) <= 0:
            price = refreshed.get(symbol)
        if price is None or float(price or 0.0) <= 0:
            print(f"[PAPER SHUTDOWN ERROR] {symbol}: no valid exit price; position not falsely settled.")
            continue
        try:
            close_position(symbol, float(price), "PAPER BOT SHUTDOWN SETTLEMENT")
        except Exception as exc:
            print(f"[PAPER SHUTDOWN ERROR] {symbol}: settlement failed: {exc}")


# ============================================================
# MAIN
# ============================================================

def main():

    global account_balance, account_available_balance

    load_persistent_state()
    ai_load_state()
    ai_start_shadow_thread()
    start_seed_worker()
    start_api_monitor()
    print(f"Connecting to {WS_URL} ...")

    if not API_KEY or not SECRET_KEY:
        print(
            "[WARNING] API_KEY / SECRET_KEY not set - REST calls "
            "(watchlist + ticker) will fail. Set COINSWITCH_API_KEY "
            "and COINSWITCH_SECRET_KEY env vars, or edit the config "
            "at the top of this file."
        )

    if LIVE_MODE and ORDERS_ENABLED:

        print()
        print("!" * 68)
        print("!!! LIVE TRADING IS ENABLED !!!")
        print("!" * 68)
        print(f"Real MARKET orders WILL be placed at {TRADE_LEVERAGE}x leverage")
        print("with REAL money on your CoinSwitch PRO futures account.")
        print("Type YES (all caps) to continue, anything else to abort.")
        print("!" * 68)

        confirmation = input("> ").strip()

        if confirmation != "YES":
            print("Aborted. LIVE_MODE remains disabled.")
            return

        print("Confirmed. Starting live trading...")

    elif PAPER_MODE and ORDERS_ENABLED:

        print()
        print("=" * 68)
        print("PAPER TRADING ENABLED")
        print(f"Paper Capital : {paper_capital:.6f} USDT")
        print("No real CoinSwitch orders will be placed.")
        print("=" * 68)

    else:

        print()
        print("=" * 68)
        print("ORDERS DISABLED")
        print(f"PAPER_MODE = {PAPER_MODE}")
        print(f"LIVE_MODE  = {LIVE_MODE}")
        print("No orders or simulated trades will be executed.")
        print("=" * 68)

    try:

        sio.connect(
            WS_URL,
            namespaces=[NAMESPACE],
            socketio_path=SOCKETIO_PATH,
            transports=["websocket"],
        )

    except Exception as e:

        print(f"[Connection Error] {e}")
        return

    if LIVE_MODE:
        refresh_account_balance()
    else:
        with state_lock:
            account_balance = None
            account_available_balance = None

    load_hft_watchlist(force=True)

    # LIVE exchange reconciliation MUST happen before scanner threads
    # are allowed to create new positions.
    if LIVE_MODE and ORDERS_ENABLED:
        recover_live_positions()

        start_protection_watchdog_thread()

        start_live_position_sync_thread()

    balance_thread = threading.Thread(target=balance_poll_loop, daemon=True)
    balance_thread.start()

    scanner_thread = threading.Thread(target=hft_scanner_loop, daemon=True)
    scanner_thread.start()

    display_thread = threading.Thread(
        target=display_loop,
        daemon=True,
        name="display_loop",
    )
    display_thread.start()

    ticker_thread = threading.Thread(target=ticker_poll_loop, daemon=True)
    ticker_thread.start()

    watchlist_thread = threading.Thread(target=watchlist_refresh_loop, daemon=True)
    watchlist_thread.start()

    try:

        sio.wait()

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        try:
            _settle_open_paper_positions_on_shutdown()
        except Exception as exc:
            print(f"[PAPER SHUTDOWN ERROR] settlement handler failed: {exc}")

        try:
            sio.disconnect()
        except Exception:
            pass


# ============================================================
# LIVE EXCHANGE READ-ONLY RECONCILIATION
# CoinSwitch Futures
# NO ORDER PLACEMENT
# NO ORDER CANCELLATION
# ============================================================

def get_live_order_status(order_id):
    """Read-only: fetch one futures order by CoinSwitch order_id."""
    if not order_id:
        return None

    try:
        path = "/trade/api/v2/futures/order"
        headers, signed_path = sign_request(
            "GET",
            path,
            params={"order_id": str(order_id)},
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE ORDER STATUS] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE ORDER STATUS ERROR] {exc}")
        return None


def _get_all_live_positions_with_status():
    """Read-only futures position request for all open positions."""
    try:
        path = "/trade/api/v2/futures/positions"
        headers, signed_path = sign_request(
            "GET",
            path,
            params={
                "exchange": "EXCHANGE_2",
            },
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE POSITIONS BULK] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()

        # Do not spam the log with the normal empty-position response.
        if not (
            isinstance(payload, dict)
            and str(payload.get("message") or "").strip().lower()
            == "there are no open positions"
        ):
            print(json.dumps(payload, indent=2))

        return True, payload

    except Exception as exc:
        print(f"[LIVE POSITIONS BULK ERROR] {exc}")
        return False, None


def _get_live_positions_with_status(symbol):
    """Read-only futures position request with explicit API success state."""
    if not symbol:
        return False, None

    try:
        symbol = str(symbol).upper()

        path = "/trade/api/v2/futures/positions"
        headers, signed_path = sign_request(
            "GET",
            path,
            params={
                "exchange": "EXCHANGE_2",
                "symbol": symbol,
            },
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE POSITIONS] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()

        # Do not spam the log with the normal empty-position response.
        # The payload is still returned to the recovery/sync logic.
        if not (
            isinstance(payload, dict)
            and str(payload.get("message") or "").strip().lower()
            == "there are no open positions"
        ):
            print(json.dumps(payload, indent=2))

        return True, payload

    except Exception as exc:
        print(f"[LIVE POSITIONS ERROR] {symbol}: {exc}")
        return False, None


def get_live_positions(symbol):
    """Read-only: fetch currently open futures position for a symbol."""
    if not symbol:
        return None

    try:
        symbol = str(symbol).upper()

        path = "/trade/api/v2/futures/positions"
        headers, signed_path = sign_request(
            "GET",
            path,
            params={
                "exchange": "EXCHANGE_2",
                "symbol": symbol,
            },
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE POSITIONS] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE POSITIONS ERROR] {symbol}: {exc}")
        return None


def get_live_open_orders(symbol=None, limit=20):
    """
    Read-only: fetch currently working (non-terminal) futures orders.

    Official endpoint: POST /trade/api/v2/futures/orders/open
    (NOT "GET /trade/api/v2/futures/orders" - that path doesn't exist
    on the Futures v2 API; Open Orders is a POST with a JSON body,
    mirroring Closed Orders below.)
    """
    try:
        path = "/trade/api/v2/futures/orders/open"

        body = {
            "exchange": "EXCHANGE_2",
            "limit": int(limit),
        }

        if symbol:
            body["symbol"] = str(symbol).upper()

        headers, signed_path = sign_request(
            "POST",
            path,
        )

        response = cs_post(
            BASE_URL + signed_path,
            headers=headers,
            json=body,
            timeout=15,
        )

        print(f"[LIVE OPEN ORDERS] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE OPEN ORDERS ERROR] {exc}")
        return None


def get_live_closed_orders(symbol=None, limit=20):
    """Read-only: fetch recently closed futures orders."""
    try:
        path = "/trade/api/v2/futures/orders/closed"

        body = {
            "exchange": "EXCHANGE_2",
            "limit": int(limit),
        }

        if symbol:
            body["symbol"] = str(symbol).upper()

        headers, signed_path = sign_request(
            "POST",
            path,
        )

        response = cs_post(
            BASE_URL + signed_path,
            headers=headers,
            json=body,
            timeout=15,
        )

        print(f"[LIVE CLOSED ORDERS] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE CLOSED ORDERS ERROR] {exc}")
        return None


def get_live_symbol_leverage(symbol):
    """Read-only: fetch exchange-confirmed leverage for a symbol."""
    if not symbol:
        return None

    try:
        symbol = str(symbol).upper()

        path = "/trade/api/v2/futures/leverage"

        headers, signed_path = sign_request(
            "GET",
            path,
            params={
                "exchange": "EXCHANGE_2",
                "symbol": symbol,
            },
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE LEVERAGE] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE LEVERAGE ERROR] {symbol}: {exc}")
        return None


# ============================================================
# ADDITIONAL OFFICIAL ENDPOINTS (previously missing from this file)
#
# These are NOT wired into the automatic strategy loop - the bot
# never calls them on its own. They're here as ready-to-use manual
# utilities matching the official Futures v2 reference. Cancel Order
# / Cancel All / Add Margin place or modify real state when
# ORDERS_ENABLED + LIVE_MODE are on - use them deliberately.
# ============================================================

def cancel_order(order_id):
    """
    DELETE /trade/api/v2/futures/order

    Cancel a single open futures order by order_id. Returns the
    parsed response dict on success, or None on any failure.
    """
    if not order_id:
        return None

    try:
        headers, signed_path = sign_request(
            "DELETE",
            "/trade/api/v2/futures/order",
            params={"order_id": str(order_id)},
        )

        response = cs_delete(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[CANCEL ORDER] {order_id} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[CANCEL ORDER ERROR] {order_id}: {exc}")
        return None


def cancel_all_open_orders(symbol=None):
    """
    POST /trade/api/v2/futures/cancel_all

    Cancel every open order (optionally scoped to one symbol).
    Returns the parsed response dict (with "orders_ids") on
    success, or None on any failure.
    """
    try:
        body = {"exchange": "EXCHANGE_2"}
        if symbol:
            body["symbol"] = str(symbol).upper()

        headers, signed_path = sign_request(
            "POST",
            "/trade/api/v2/futures/cancel_all",
        )

        response = cs_post(
            BASE_URL + signed_path,
            headers=headers,
            json=body,
            timeout=15,
        )

        print(f"[CANCEL ALL] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[CANCEL ALL ERROR] {exc}")
        return None


def add_margin(symbol, margin):
    """
    POST /trade/api/v2/futures/add_margin

    Top up margin (in USDT) on an open position, pushing the
    liquidation price further from the mark price. `margin` must
    be <= available wallet balance. Returns the parsed response
    dict on success, or None on any failure.
    """
    if not symbol:
        return None

    try:
        body = {
            "exchange": "EXCHANGE_2",
            "symbol": str(symbol).upper(),
            "margin": margin,
        }

        headers, signed_path = sign_request(
            "POST",
            "/trade/api/v2/futures/add_margin",
        )

        response = cs_post(
            BASE_URL + signed_path,
            headers=headers,
            json=body,
            timeout=15,
        )

        print(f"[ADD MARGIN] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[ADD MARGIN ERROR] {symbol}: {exc}")
        return None


def get_live_transactions(symbol=None, limit=20):
    """
    GET /trade/api/v2/futures/transactions

    Read-only: fees, funding payments, realized PnL, and add-margin
    history. Returns the parsed response dict, or None on failure.
    """
    try:
        params = {"exchange": "EXCHANGE_2", "limit": int(limit)}
        if symbol:
            params["symbol"] = str(symbol).upper()

        headers, signed_path = sign_request(
            "GET",
            "/trade/api/v2/futures/transactions",
            params=params,
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE TRANSACTIONS] HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE TRANSACTIONS ERROR] {exc}")
        return None


def get_live_order_book(symbol):
    """
    GET /trade/api/v2/futures/order_book

    Read-only: current bids/asks snapshot for a symbol. Returns
    the parsed response dict, or None on failure.
    """
    if not symbol:
        return None

    try:
        symbol = str(symbol).upper()

        headers, signed_path = sign_request(
            "GET",
            "/trade/api/v2/futures/order_book",
            params={"exchange": "EXCHANGE_2", "symbol": symbol},
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE ORDER BOOK] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE ORDER BOOK ERROR] {symbol}: {exc}")
        return None


def get_live_ticker(symbol):
    """
    GET /trade/api/v2/futures/ticker

    Read-only: 24h stats + funding info for ONE symbol (unlike
    fetch_all_pairs_ticker(), which covers every symbol in one
    call). Returns the parsed response dict, or None on failure.
    """
    if not symbol:
        return None

    try:
        symbol = str(symbol).upper()

        headers, signed_path = sign_request(
            "GET",
            "/trade/api/v2/futures/ticker",
            params={"exchange": "EXCHANGE_2", "symbol": symbol},
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE TICKER] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE TICKER ERROR] {symbol}: {exc}")
        return None


def get_live_trades(symbol):
    """
    GET /trade/api/v2/futures/trades

    Read-only: recent public trade prints for a symbol. Returns
    the parsed response dict, or None on failure.
    """
    if not symbol:
        return None

    try:
        symbol = str(symbol).upper()

        headers, signed_path = sign_request(
            "GET",
            "/trade/api/v2/futures/trades",
            params={"exchange": "EXCHANGE_2", "symbol": symbol},
        )

        response = cs_get(
            BASE_URL + signed_path,
            headers=headers,
            timeout=15,
        )

        print(f"[LIVE TRADES] {symbol} HTTP: {response.status_code}")
        response.raise_for_status()

        payload = response.json()
        print(json.dumps(payload, indent=2))
        return payload

    except Exception as exc:
        print(f"[LIVE TRADES ERROR] {symbol}: {exc}")
        return None


def live_exchange_audit(symbol="BTCUSDT"):
    """
    Complete read-only LIVE exchange audit.

    IMPORTANT:
    This function NEVER places or cancels an order.
    """

    print()
    print("=" * 78)
    print("COINSWITCH LIVE EXCHANGE AUDIT")
    print("=" * 78)
    print(f"SYMBOL: {symbol}")
    print()

    print("-" * 78)
    print("1. FUTURES WALLET BALANCE")
    print("-" * 78)
    try:
        balance = fetch_futures_balance()
        print(json.dumps(balance, indent=2))
    except Exception as exc:
        print(f"[BALANCE ERROR] {exc}")

    print()
    print("-" * 78)
    print("2. OPEN POSITION")
    print("-" * 78)
    get_live_positions(symbol)

    print()
    print("-" * 78)
    print("3. OPEN ORDERS")
    print("-" * 78)
    get_live_open_orders(symbol)

    print()
    print("-" * 78)
    print("4. CLOSED ORDERS")
    print("-" * 78)
    get_live_closed_orders(symbol, limit=20)

    print()
    print("-" * 78)
    print("5. EXCHANGE LEVERAGE")
    print("-" * 78)
    get_live_symbol_leverage(symbol)

    print()
    print("=" * 78)
    print("LIVE EXCHANGE AUDIT COMPLETE")
    print("READ-ONLY: NO ORDER WAS PLACED OR CANCELLED")
    print("=" * 78)
    print()


if __name__ == "__main__" and os.environ.get("COINSWITCH_AUDIT_ONLY") == "1":
    live_exchange_audit(os.environ.get("COINSWITCH_AUDIT_SYMBOL", "BTCUSDT"))

# ============================================================
# PROGRAM ENTRY
# All function definitions must exist before main() executes.
# ============================================================

if __name__ == "__main__":
    main()
