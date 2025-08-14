import re
import requests
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, Div, Slider, Select, HoverTool, CustomJS
)
from bokeh.layouts import column, row
from bokeh.io import output_file
from bokeh.palettes import Category10

# ---------------------------------------
# Configuration
# ---------------------------------------
SERIES_TICKER = "KXHIGHNY"
STATUS = "settled"
BASE_API = "https://api.elections.kalshi.com/trade-api/v2"
EVENTS_API_MAX_LIMIT = 200

# Time window: previous day 12:00 PM ET → event day 9:00 PM ET
LOCAL_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ_NAME = "America/New_York"
START_PREV_DAY_HOUR_ET = 12
START_PREV_DAY_MIN_ET  = 0
END_DAY_HOUR_ET        = 21
END_DAY_MIN_ET         = 0

# Date range settings
FIRST_AVAILABLE_DATE = datetime(2024, 10, 25, tzinfo=LOCAL_TZ)  # First day temp markets available
LOOKBACK_DAYS = None  # None = use all data from FIRST_AVAILABLE_DATE; or set integer days

# Cache settings
CACHE_FILE = f"kalshi_{SERIES_TICKER}_cache.pkl"
FORCE_REFRESH = False  # True to force re-download
EXPORT_CSV = False     # True to export CSV (can be large!)

# Calibration settings
DEFAULT_BINS = 10
BIN_OPTIONS = ["5", "10", "20", "50", "100"]

# ---------------------------------------
# Helpers
# ---------------------------------------
def generate_event_tickers():
    """Generate event tickers from FIRST_AVAILABLE_DATE to yesterday."""
    today = datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)

    if LOOKBACK_DAYS is None:
        start_date = FIRST_AVAILABLE_DATE
    else:
        lookback_start = today - timedelta(days=LOOKBACK_DAYS)
        start_date = max(FIRST_AVAILABLE_DATE, lookback_start)

    current_date = start_date
    tickers = []

    while current_date <= yesterday:
        yy = str(current_date.year)[2:]            # last 2 digits
        mmm = current_date.strftime("%b").upper()  # 3-letter month
        dd = f"{current_date.day:02d}"
        tickers.append(f"{SERIES_TICKER}-{yy}{mmm}{dd}")
        current_date += timedelta(days=1)

    total_days = (yesterday - start_date).days + 1
    print(f"Generated {len(tickers)} event tickers from {start_date.strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')} ({total_days} days)")
    return tickers

MON_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
TICKER_DATE_RE = re.compile(r".*?-(\d{2})([A-Z]{3})(\d{2})$")

def fetch_event_list(series_ticker: str, status: str, limit: int = EVENTS_API_MAX_LIMIT) -> list[dict]:
    limit = min(limit, EVENTS_API_MAX_LIMIT)
    url = f"{BASE_API}/events"
    params = {"series_ticker": series_ticker, "status": status, "limit": str(limit)}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("events", [])

def fetch_markets_for_event(event_ticker: str) -> list[str]:
    url = f"{BASE_API}/markets"
    params = {"event_ticker": event_ticker}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return [m.get("ticker") for m in r.json().get("markets", []) if isinstance(m, dict) and m.get("ticker")]

def parse_event_date_from_ticker(event_ticker: str):
    m = TICKER_DATE_RE.match(event_ticker or "")
    if not m: return None
    yy, mon_str, dd = m.groups()
    mon = MON_MAP.get(mon_str.upper())
    if not mon: return None
    try:
        return datetime(2000+int(yy), mon, int(dd), tzinfo=LOCAL_TZ)
    except ValueError:
        return None

def event_window_prevnoon_to_9pm_unix(event_ticker: str):
    base_dt = parse_event_date_from_ticker(event_ticker)
    if base_dt is None: return None
    prev_day = (base_dt - timedelta(days=1)).replace(
        hour=START_PREV_DAY_HOUR_ET, minute=START_PREV_DAY_MIN_ET, second=0, microsecond=0
    )
    end_local = base_dt.replace(
        hour=END_DAY_HOUR_ET, minute=END_DAY_MIN_ET, second=0, microsecond=0
    )
    return int(prev_day.astimezone(timezone.utc).timestamp()), int(end_local.astimezone(timezone.utc).timestamp())

def fetch_mid_df(event_ticker: str, market_ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    params = {"start_ts": str(start_ts), "end_ts": str(end_ts), "period_interval": "1"}
    paths = [
        f"{BASE_API}/series/{event_ticker}/markets/{market_ticker}/candlesticks",
        f"{BASE_API}/markets/{market_ticker}/candlesticks",
    ]
    for url in paths:
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200: continue
            j = r.json()
            candles = j.get("candlesticks", [])
            if not candles:
                for alt in ("5","15","60"):
                    r2 = requests.get(url, params={**params, "period_interval": alt}, timeout=20)
                    if r2.status_code != 200: continue
                    c2 = r2.json().get("candlesticks", [])
                    if c2: candles = c2; break
            if not candles: continue

            df = pd.json_normalize(candles)
            if "end_period_ts" not in df.columns: continue
            dt_utc = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
            dt_local = dt_utc.dt.tz_convert(DISPLAY_TZ_NAME)
            df["dt_plot"] = dt_local
            if "yes_bid.close" in df.columns and "yes_ask.close" in df.columns:
                df["mid"] = (df["yes_bid.close"] + df["yes_ask.close"]) / 2.0
            elif "yes_bid" in df.columns and "yes_ask" in df.columns:
                df["mid"] = (df["yes_bid"] + df["yes_ask"]) / 2.0
            elif "last_price" in df.columns:
                df["mid"] = df["last_price"]
            else:
                return pd.DataFrame()
            df = df[df["mid"].notna()].sort_values("dt_plot")
            return df[["dt_plot","mid"]]
        except requests.RequestException:
            continue
    return pd.DataFrame()

def fetch_market_outcome(market_ticker: str) -> int | None:
    """Fetch final outcome: 1 if YES won, 0 if NO won."""
    url = f"{BASE_API}/markets/{market_ticker}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        if not isinstance(j, dict):
            return None
    except requests.RequestException:
        return None

    sp = j.get("settlement_price")
    result = j.get("result")

    if "market" in j and isinstance(j["market"], dict):
        market_data = j["market"]
        if sp is None:
            sp = market_data.get("settlement_price")
        if result is None:
            result = market_data.get("result")

    if result == "yes":
        return 1
    if result == "no":
        return 0

    if sp is not None:
        try:
            sp_float = float(sp)
            return 1 if sp_float >= 50 else 0
        except Exception:
            pass

    return None

def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Cache saved to {filename}")

def load_cache(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Cache loaded from {filename}")
            return data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None

def export_to_csv(calibration_data, filename):
    print(f"Preparing CSV export with {len(calibration_data)} markets...")
    total_rows = sum(len(data['probs']) for data in calibration_data)
    estimated_size_mb = total_rows * 8 * 7 / (1024 * 1024)
    print(f"Estimated CSV size: ~{estimated_size_mb:.1f} MB ({total_rows:,} rows)")
    if estimated_size_mb > 100:
        print("Warning: Large file size. Consider setting EXPORT_CSV = False")
    rows = []
    for i, data in enumerate(calibration_data):
        if i % 50 == 0:
            print(f"Processing market {i+1}/{len(calibration_data)}")
        for time_min, prob in zip(data['time_of_day_minutes'], data['probs']):
            rows.append({
                'event': data['event'],
                'market': data['market'],
                'time_of_day_minutes': time_min,
                'probability': prob,
                'outcome': data['outcome'],
                'category': data['category'],
                'avg_prob': data['avg_prob']
            })
    df = pd.DataFrame(rows)
    try:
        df.to_csv(filename, index=False)
        print(f"✓ Data exported to {filename} ({len(df):,} rows)")
    except OSError as e:
        print(f"✗ Failed to export CSV: {e}")
        return None
    return df

# ---------------------------------------
# Build calibration dataset with caching
# ---------------------------------------
def build_calibration_dataset():
    if not FORCE_REFRESH:
        cached_data = load_cache(CACHE_FILE)
        if cached_data is not None:
            print(f"Using cached data with {len(cached_data)} markets")
            return cached_data

    print("Building fresh calibration dataset...")
    event_tickers = generate_event_tickers()
    print(f"Loading calibration data for {len(event_tickers)} day(s)")

    calibration_data = []
    yes_count, no_count = 0, 0

    for i, etkr in enumerate(event_tickers):
        print(f"Processing {i+1}/{len(event_tickers)}: {etkr}")

        win = event_window_prevnoon_to_9pm_unix(etkr)
        if win is None:
            continue
        start_ts, end_ts = win

        try:
            mkts = fetch_markets_for_event(etkr)
        except requests.RequestException:
            continue

        for mkt in mkts:
            df = fetch_mid_df(etkr, mkt, start_ts, end_ts)
            if df.empty:
                continue

            outcome = fetch_market_outcome(mkt)
            if outcome is None:
                continue

            yes_count += int(outcome == 1)
            no_count  += int(outcome == 0)

            df["prob"] = (df["mid"] / 100.0).clip(0.001, 0.999)

            # Convert to time-of-day (minutes since start of window)
            window_start_dt = pd.Timestamp.fromtimestamp(start_ts, tz='UTC').tz_convert(DISPLAY_TZ_NAME)
            df["time_of_day_minutes"] = ((df["dt_plot"] - window_start_dt).dt.total_seconds() / 60).astype(int)

            # Optional category by avg probability
            avg_prob = df["prob"].mean()
            if   avg_prob < 0.2: category = "very_low"
            elif avg_prob < 0.4: category = "low"
            elif avg_prob < 0.6: category = "mid"
            elif avg_prob < 0.8: category = "high"
            else:                category = "very_high"

            calibration_data.append({
                "event": etkr,
                "market": mkt,
                "time_of_day_minutes": df["time_of_day_minutes"].values,
                "probs": df["prob"].values,
                "outcome": int(outcome),
                "category": category,
                "avg_prob": float(avg_prob)
            })

    print(f"Loaded {len(calibration_data)} markets: {yes_count} YES, {no_count} NO outcomes")
    save_cache(calibration_data, CACHE_FILE)

    if EXPORT_CSV:
        csv_filename = f"kalshi_{SERIES_TICKER}_data.csv"
        export_to_csv(calibration_data, csv_filename)
    else:
        print("Skipping CSV export (set EXPORT_CSV = True to enable)")

    return calibration_data

# ---------------------------------------
# Load and prepare data
# ---------------------------------------
print("Building calibration dataset...")
calibration_data = build_calibration_dataset()
if not calibration_data:
    raise SystemExit("No calibration data loaded.")
print(f"Loaded {len(calibration_data)} market time series")

# Time range - minutes from window start
all_times_minutes = []
for data in calibration_data:
    all_times_minutes.extend(data["time_of_day_minutes"])
min_time_minutes = min(all_times_minutes)
max_time_minutes = max(all_times_minutes)
print(f"Time range: {min_time_minutes} to {max_time_minutes} minutes from window start")

def minutes_to_time_label(minutes: int) -> str:
    """
    Minutes since Prev Day 12:00 PM ET -> human label.
    0 -> Prev Day 12:00 PM; 720 -> Event Day 12:00 AM.
    """
    m = max(0, int(round(minutes)))
    base_min = 12 * 60  # 12:00 PM
    total_min = base_min + m
    day_offset = total_min // (24 * 60)             # 0 prev day, 1 event day
    minute_of_day = total_min % (24 * 60)
    hh24 = minute_of_day // 60
    mm = minute_of_day % 60
    am_pm = "AM" if hh24 < 12 else "PM"
    hh12 = ((hh24 + 11) % 12) + 1
    day_label = "Event Day" if day_offset >= 1 else "Prev Day"
    return f"{day_label} {hh12}:{mm:02d} {am_pm}"

initial_time_minutes = min_time_minutes + int((max_time_minutes - min_time_minutes) * 0.75)

# ---------------------------------------
# Bokeh sources and controls
# ---------------------------------------
diagonal_source = ColumnDataSource(data=dict(x=[0, 1], y=[0, 1]))
calibration_source = ColumnDataSource(data=dict(x=[], y=[], size=[], count=[], color=[], line_color=[]))
individual_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))

# Added sources for base-rate line and uncertainty band
base_rate_source = ColumnDataSource(data=dict(x=[0, 0], y0=[0, 0], y1=[1, 1]))
band_source = ColumnDataSource(data=dict(xs=[], lower=[], upper=[]))

time_slider = Slider(
    title="Time of Day (minutes since Prev Day 12:00 PM ET)",
    start=min_time_minutes,
    end=max_time_minutes,
    value=initial_time_minutes,
    step=15,
    width=600,
    format="0"
)
bins_select = Select(title="Bins", value=str(DEFAULT_BINS), options=BIN_OPTIONS, width=100)
category_options = ["all", "very_low", "low", "mid", "high", "very_high"]
category_select = Select(title="Category", value="all", options=category_options, width=120)

# ---------------------------------------
# Figure
# ---------------------------------------
p = figure(
    width=900, height=650,
    x_range=(0, 1), y_range=(0, 1),
    title="Kalshi Market Calibration Analysis",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

# Perfect calibration diagonal
p.line("x", "y", source=diagonal_source,
       line_color="gray", line_dash="dashed", line_width=2, alpha=0.7,
       legend_label="Perfect calibration")

# Uncertainty band (two patches: lower and upper polygons)
p.patch("xs", "lower", source=band_source, color="#1f77b4", alpha=0.15, line_alpha=0, legend_label="Calibration uncertainty")
p.patch("xs", "upper", source=band_source, color="#1f77b4", alpha=0.15, line_alpha=0)

# Base rate vertical line
p.segment(x0="x", y0="y0", x1="x", y1="y1",
          source=base_rate_source,
          line_color="#1f77b4", line_dash="dashed", line_width=2,
          legend_label="Base event rate")

# Bin averages
calibration_glyph = p.scatter(
    "x", "y",
    marker="circle",
    size="size",
    source=calibration_source,
    fill_color="color",
    line_color="line_color",
    alpha=0.8,
    legend_label="Bin averages"
)

# Individual outcome points (rug)
p.scatter(
    "x", "y",
    marker="circle",
    source=individual_source,
    size=3,
    fill_color="color",
    line_color=None,
    alpha=0.12
)

p.xaxis.axis_label = "Predicted Probability (Market Price)"
p.yaxis.axis_label = "Observed Frequency (YES)"
p.legend.location = "top_left"

hover = HoverTool(
    tooltips=[("Predicted", "@x{0.000}"), ("Observed", "@y{0.000}"), ("Count", "@count")],
    renderers=[calibration_glyph]
)
p.add_tools(hover)

stats_div = Div(text="", width=900)

# ---------------------------------------
# JS data and callback
# ---------------------------------------
js_data = {
    'calibration_data': [
        {
            'time_of_day_minutes': data['time_of_day_minutes'].tolist(),
            'probs': data['probs'].tolist(),
            'outcome': int(data['outcome']),
            'category': data['category']
        }
        for data in calibration_data
    ],
    'calib_color': Category10[10],
    'line_color': "#333333",
    'rug_color': "black"
}

callback_js = CustomJS(
    args=dict(
        time_slider=time_slider,
        bins_select=bins_select,
        category_select=category_select,
        calibration_source=calibration_source,
        individual_source=individual_source,
        stats_div=stats_div,
        js_data=js_data,
        base_rate_source=base_rate_source,
        band_source=band_source
    ),
    code="""
    // Formatter: anchor at Prev Day 12:00 PM
    function minutes_to_time_label(minutes) {
        const m = Math.max(0, Math.round(minutes));
        const baseMin = 12 * 60; // 12:00 PM
        const totalMin = baseMin + m;
        const dayOffset = Math.floor(totalMin / (24 * 60));
        const minuteOfDay = totalMin % (24 * 60);
        const hh24 = Math.floor(minuteOfDay / 60);
        const mm = String(minuteOfDay % 60).padStart(2, '0');
        const ampm = hh24 < 12 ? "AM" : "PM";
        const hh12 = ((hh24 + 11) % 12) + 1;
        const dayLabel = dayOffset === 0 ? "Prev Day" : "Event Day";
        return `${dayLabel} ${hh12}:${mm} ${ampm}`;
    }

    function update_calibration() {
        const current_time_minutes = time_slider.value;
        const n_bins = parseInt(bins_select.value);
        const category_filter = category_select.value;

        const preds = [];
        const outs  = [];

        // Gather snapshot predictions/outcomes
        for (const data of js_data.calibration_data) {
            if (category_filter !== "all" && data.category !== category_filter) continue;
            let last_idx = -1;
            const time_mins = data.time_of_day_minutes;
            for (let i = 0; i < time_mins.length; i++) {
                if (time_mins[i] <= current_time_minutes) last_idx = i;
                else break;
            }
            if (last_idx >= 0) {
                preds.push(data.probs[last_idx]);
                outs.push(data.outcome);
            }
        }

        if (preds.length === 0) {
            calibration_source.data = {x: [], y: [], size: [], count: [], color: [], line_color: []};
            individual_source.data   = {x: [], y: [], color: []};
            base_rate_source.data    = {x: [0,0], y0: [0,0], y1: [1,1]};
            band_source.data         = {xs: [], lower: [], upper: []};
            stats_div.text = "No data available for current selection";
            return;
        }

        // Bin edges
        const edges = [];
        for (let i = 0; i <= n_bins; i++) edges.push(i / n_bins);

        const bin_pred = [];
        const bin_obs  = [];
        const bin_counts = [];

        for (let b = 0; b < n_bins; b++) {
            const lo = edges[b], hi = edges[b+1];
            const pbin = [], obin = [];
            for (let j = 0; j < preds.length; j++) {
                const p = preds[j];
                if ((b === n_bins - 1 && p >= lo && p <= hi) || (p >= lo && p < hi)) {
                    pbin.push(p);
                    obin.push(outs[j]);
                }
            }
            if (pbin.length > 0) {
                const mp = pbin.reduce((a,b)=>a+b,0)/pbin.length;
                const mo = obin.reduce((a,b)=>a+b,0)/obin.length;
                bin_pred.push(mp);
                bin_obs.push(mo);
                bin_counts.push(pbin.length);
            }
        }

        // Point sizes and colors for bins
        const sizes = bin_counts.map(c => Math.max(8, Math.min(30, 8 + 3 * Math.log(c))));
        const colors = Array(bin_pred.length).fill(js_data.calib_color);
        const lines  = Array(bin_pred.length).fill(js_data.line_color);

        calibration_source.data = {
            x: bin_pred, y: bin_obs, size: sizes, count: bin_counts, color: colors, line_color: lines
        };

        // Individual (rug) points
        individual_source.data = {
            x: preds, y: outs, color: Array(preds.length).fill(js_data.rug_color)
        };

        // ---- Base rate vertical line (overall YES frequency among included markets) ----
        const base_rate = outs.reduce((s,o)=>s+o,0) / outs.length;
        base_rate_source.data = { x: [base_rate, base_rate], y0: [0,0], y1: [1,1] };

        // ---- Uncertainty band around y=x (95% normal approx) ----
        // For smooth ribbon, use 0..1 grid; nearest bin_count defines n for variance
        const xs = [];
        const lower = [];
        const upper = [];
        const steps = 121;
        for (let i = 0; i < steps; i++) {
            const x = i / (steps - 1);
            let best_k = -1, best_d = 1e9;
            for (let k = 0; k < bin_pred.length; k++) {
                const d = Math.abs(x - bin_pred[k]);
                if (d < best_d) { best_d = d; best_k = k; }
            }
            const n = best_k >= 0 ? bin_counts[best_k] : 0;
            const p = x;
            let lo = p, hi = p;
            if (n > 0) {
                const se = Math.sqrt(Math.max(p * (1 - p), 1e-9) / n);
                const z = 1.96;
                lo = Math.max(0, p - z * se);
                hi = Math.min(1, p + z * se);
            }
            xs.push(x);
            lower.push(lo);
            upper.push(hi);
        }
        band_source.data = { xs: xs, lower: lower, upper: upper };

        // ---- Stats (Brier & Skill) ----
        const brier = preds.reduce((s,p,i)=> s + Math.pow(p - outs[i], 2), 0) / preds.length;
        const ref   = outs.reduce((s,o)=> s + Math.pow(base_rate - o, 2), 0) / outs.length;
        const skill = ref > 0 ? 1 - (brier / ref) : NaN;

        const time_str = minutes_to_time_label(current_time_minutes);
        stats_div.text = `<b>Time:</b> ${time_str} | ` +
                         `<b>Markets:</b> ${preds.length} | ` +
                         `<b>Brier Score:</b> ${brier.toFixed(3)} | ` +
                         `<b>Skill Score:</b> ${isNaN(skill) ? 'N/A' : skill.toFixed(3)}`;
    }

    // Initial draw
    update_calibration();
    this.update_calibration = update_calibration;
    """
)

# Attach callbacks
time_slider.js_on_change('value', callback_js)
bins_select.js_on_change('value', callback_js)
category_select.js_on_change('value', callback_js)

# ---------------------------------------
# Layout and output
# ---------------------------------------
header = Div(
    text=f"<h2>{SERIES_TICKER} Calibration Analysis</h2>"
         f"<p>Interactive calibration across {len(calibration_data)} markets over {len(set(d['event'] for d in calibration_data))} days "
         f"(window: prev day 12:00 PM → event day 9:00 PM ET). Base rate line and calibration uncertainty band update with filters and time.</p>",
    width=900
)
controls = row(bins_select, category_select, time_slider)
layout = column(header, controls, p, stats_div, sizing_mode="scale_width")

output_file(f"kalshi_{SERIES_TICKER}_calibration.html", title=f"{SERIES_TICKER} Calibration Analysis")
print("Creating calibration plot...")
show(layout)

print("\n=== FILES CREATED ===")
print(f"✓ Cache file: {CACHE_FILE}")
if EXPORT_CSV:
    print(f"✓ CSV export: kalshi_{SERIES_TICKER}_data.csv")
else:
    print(f"- CSV export: Disabled (set EXPORT_CSV = True to enable)")
print(f"✓ HTML plot: kalshi_{SERIES_TICKER}_calibration.html")
print("\nConfiguration:")
print(f"- Date range: {FIRST_AVAILABLE_DATE.strftime('%Y-%m-%d')} to yesterday")
print(f"- Using {'ALL available data' if LOOKBACK_DAYS is None else f'lookback {LOOKBACK_DAYS} days'}")
print(f"- Total markets: {len(calibration_data)}")
print(f"- Unique days: {len(set(d['event'] for d in calibration_data))}")
print("\nTo force refresh data, set FORCE_REFRESH = True")
print("To export CSV data, set EXPORT_CSV = True (warning: can be large!)")
