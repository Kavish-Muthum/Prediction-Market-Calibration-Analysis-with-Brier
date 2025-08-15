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
    ColumnDataSource, Div, Slider, Select, HoverTool, CustomJS, CheckboxGroup
)
from bokeh.layouts import column, row
from bokeh.io import output_file
from bokeh.palettes import Category10, Set3

# ---------------------------------------
# Configuration
# ---------------------------------------
# Multiple cities to analyze
CITY_CONFIGS = {
    "NYC": {"ticker": "KXHIGHNY", "name": "New York City", "color": "#1f77b4"},
    "LAX": {"ticker": "KXHIGHLAX", "name": "Los Angeles", "color": "#ff7f0e"}, 
    "PHIL": {"ticker": "KXHIGHPHIL", "name": "Philadelphia", "color": "#2ca02c"},
    "AUS": {"ticker": "KXHIGHAUS", "name": "Austin", "color": "#d62728"},
    "MIA": {"ticker": "KXHIGHMIA", "name": "Miami", "color": "#9467bd"},
    "DEN": {"ticker": "KXHIGHDEN", "name": "Denver", "color": "#8c564b"},
    "CHI": {"ticker": "KXHIGHCHI", "name": "Chicago", "color": "#e377c2"}
}

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

# Date range settings - updated for combined data
FIRST_AVAILABLE_DATE = datetime(2025, 1, 5, tzinfo=LOCAL_TZ)  # Updated date
LOOKBACK_DAYS = None  # None = use all data from FIRST_AVAILABLE_DATE; or set integer days

# Cache settings
CACHE_FILE = "kalshi_multi_city_temp_cache.pkl"
FORCE_REFRESH = False  # True to force re-download
EXPORT_CSV = False     # True to export CSV (large)

# Calibration settings
DEFAULT_BINS = 10
BIN_OPTIONS = ["5", "10", "20", "50", "100"]

# Analysis modes
ANALYSIS_MODES = ["combined", "by_city", "comparison"]
DEFAULT_MODE = "combined"

# ---------------------------------------
# Helpers
# ---------------------------------------
def generate_event_tickers(series_ticker):
    """Generate event tickers from FIRST_AVAILABLE_DATE to yesterday for a specific series."""
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
        tickers.append(f"{series_ticker}-{yy}{mmm}{dd}")
        current_date += timedelta(days=1)

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
    estimated_size_mb = total_rows * 8 * 8 / (1024 * 1024)  # Added city column
    print(f"Estimated CSV size: ~{estimated_size_mb:.1f} MB ({total_rows:,} rows)")
    if estimated_size_mb > 100:
        print("Warning: Large file size. Consider setting EXPORT_CSV = False")
    rows = []
    for i, data in enumerate(calibration_data):
        if i % 50 == 0:
            print(f"Processing market {i+1}/{len(calibration_data)}")
        for time_min, prob in zip(data['time_of_day_minutes'], data['probs']):
            rows.append({
                'city': data['city'],
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
# Build multi-city calibration dataset
# ---------------------------------------
def build_multi_city_calibration_dataset():
    if not FORCE_REFRESH:
        cached_data = load_cache(CACHE_FILE)
        if cached_data is not None:
            print(f"Using cached data with {len(cached_data)} markets across cities")
            return cached_data

    print("Building fresh multi-city calibration dataset...")
    
    all_calibration_data = []
    city_stats = {}

    for city_code, city_config in CITY_CONFIGS.items():
        print(f"\n=== Processing {city_config['name']} ({city_code}) ===")
        series_ticker = city_config['ticker']
        
        event_tickers = generate_event_tickers(series_ticker)
        print(f"Generated {len(event_tickers)} event tickers for {city_config['name']}")

        city_data = []
        yes_count, no_count = 0, 0

        for i, etkr in enumerate(event_tickers):
            if i % 20 == 0:
                print(f"Processing {city_code} {i+1}/{len(event_tickers)}: {etkr}")

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

                # Category by avg probability
                avg_prob = df["prob"].mean()
                if   avg_prob < 0.2: category = "very_low"
                elif avg_prob < 0.4: category = "low"
                elif avg_prob < 0.6: category = "mid"
                elif avg_prob < 0.8: category = "high"
                else:                category = "very_high"

                market_data = {
                    "city": city_code,
                    "city_name": city_config['name'],
                    "event": etkr,
                    "market": mkt,
                    "time_of_day_minutes": df["time_of_day_minutes"].values,
                    "probs": df["prob"].values,
                    "outcome": int(outcome),
                    "category": category,
                    "avg_prob": float(avg_prob)
                }
                
                city_data.append(market_data)
                all_calibration_data.append(market_data)

        city_stats[city_code] = {
            'name': city_config['name'],
            'markets': len(city_data),
            'yes': yes_count,
            'no': no_count,
            'base_rate': yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
        }
        
        print(f"✓ {city_config['name']}: {len(city_data)} markets, {yes_count} YES, {no_count} NO (base rate: {city_stats[city_code]['base_rate']:.3f})")

    # Summary statistics
    total_markets = len(all_calibration_data)
    total_yes = sum(d['outcome'] for d in all_calibration_data)
    total_no = total_markets - total_yes
    
    print(f"\n=== COMBINED SUMMARY ===")
    print(f"Total markets: {total_markets}")
    print(f"Total outcomes: {total_yes} YES, {total_no} NO")
    print(f"Overall base rate: {total_yes/total_markets:.3f}")
    print(f"Unique days: {len(set(d['event'] for d in all_calibration_data))}")
    
    print("\nPer-city breakdown:")
    for city_code, stats in city_stats.items():
        print(f"  {stats['name']}: {stats['markets']} markets, base rate {stats['base_rate']:.3f}")

    # Save cache with metadata
    cache_data = {
        'calibration_data': all_calibration_data,
        'city_stats': city_stats,
        'generated_at': datetime.now().isoformat()
    }
    save_cache(cache_data, CACHE_FILE)

    if EXPORT_CSV:
        csv_filename = "kalshi_multi_city_temp_data.csv"
        export_to_csv(all_calibration_data, csv_filename)
    else:
        print("Skipping CSV export (set EXPORT_CSV = True to enable)")

    return all_calibration_data

# ---------------------------------------
# Load and prepare data
# ---------------------------------------
print("Building multi-city calibration dataset...")
cached_result = load_cache(CACHE_FILE) if not FORCE_REFRESH else None

if cached_result and isinstance(cached_result, dict) and 'calibration_data' in cached_result:
    calibration_data = cached_result['calibration_data']
    city_stats = cached_result.get('city_stats', {})
else:
    calibration_data = build_multi_city_calibration_dataset()
    city_stats = {}

if not calibration_data:
    raise SystemExit("No calibration data loaded.")

print(f"Loaded {len(calibration_data)} market time series across {len(CITY_CONFIGS)} cities")

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

# Multiple sources for different cities when in comparison mode
combined_calibration_source = ColumnDataSource(data=dict(x=[], y=[], size=[], count=[], color=[], line_color=[]))
combined_individual_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
combined_base_rate_source = ColumnDataSource(data=dict(x=[0, 0], y0=[0, 0], y1=[1, 1]))
combined_band_source = ColumnDataSource(data=dict(xs=[], lower=[], upper=[]))

# City-specific sources for comparison mode
city_sources = {}
for city_code in CITY_CONFIGS.keys():
    city_sources[city_code] = {
        'calibration': ColumnDataSource(data=dict(x=[], y=[], size=[], count=[], color=[], line_color=[])),
        'individual': ColumnDataSource(data=dict(x=[], y=[], color=[])),
        'base_rate': ColumnDataSource(data=dict(x=[0, 0], y0=[0, 0], y1=[1, 1])),
        'band': ColumnDataSource(data=dict(xs=[], lower=[], upper=[]))
    }

# Controls
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
mode_select = Select(title="Analysis Mode", value=DEFAULT_MODE, options=ANALYSIS_MODES, width=150)

# City selection for filtering
city_options = [(city_code, city_config['name']) for city_code, city_config in CITY_CONFIGS.items()]
default_cities = list(range(len(city_options)))  # All cities selected by default
city_checkbox = CheckboxGroup(labels=[name for _, name in city_options], active=default_cities, width=300)

# ---------------------------------------
# Figure
# ---------------------------------------
p = figure(
    width=1000, height=700,
    x_range=(0, 1), y_range=(0, 1),
    title="Multi-City Kalshi Temperature Market Calibration Analysis",
    tools="pan,wheel_zoom,box_zoom,reset,save"
)

# Perfect calibration diagonal
p.line("x", "y", source=diagonal_source,
       line_color="gray", line_dash="dashed", line_width=2, alpha=0.7,
       legend_label="Perfect calibration")

# Combined analysis elements
p.patch("xs", "lower", source=combined_band_source, color="#1f77b4", alpha=0.15, line_alpha=0, legend_label="Calibration uncertainty")
p.patch("xs", "upper", source=combined_band_source, color="#1f77b4", alpha=0.15, line_alpha=0)

p.segment(x0="x", y0="y0", x1="x", y1="y1",
          source=combined_base_rate_source,
          line_color="#1f77b4", line_dash="dashed", line_width=2,
          legend_label="Base event rate")

combined_calibration_glyph = p.scatter(
    "x", "y", marker="circle", size="size",
    source=combined_calibration_source,
    fill_color="color", line_color="line_color", alpha=0.8,
    legend_label="Combined bin averages"
)

p.scatter("x", "y", marker="circle", source=combined_individual_source,
          size=3, fill_color="color", line_color=None, alpha=0.12)

# City-specific elements (initially hidden)
city_glyphs = {}
for city_code, city_config in CITY_CONFIGS.items():
    sources = city_sources[city_code]
    color = city_config['color']
    name = city_config['name']
    
    # City-specific uncertainty band and base rate
    p.patch("xs", "lower", source=sources['band'], color=color, alpha=0.1, line_alpha=0, visible=False)
    p.patch("xs", "upper", source=sources['band'], color=color, alpha=0.1, line_alpha=0, visible=False)
    
    p.segment(x0="x", y0="y0", x1="x", y1="y1", source=sources['base_rate'],
              line_color=color, line_dash="dashed", line_width=2, visible=False,
              legend_label=f"{name} base rate")
    
    # City calibration points
    city_glyphs[city_code] = p.scatter(
        "x", "y", marker="circle", size="size",
        source=sources['calibration'],
        fill_color=color, line_color="#333333", alpha=0.8, visible=False,
        legend_label=f"{name} bins"
    )
    
    # City individual points
    p.scatter("x", "y", marker="circle", source=sources['individual'],
              size=3, fill_color=color, line_color=None, alpha=0.08, visible=False)

p.xaxis.axis_label = "Predicted Probability (Market Price)"
p.yaxis.axis_label = "Observed Frequency (YES)"
p.legend.location = "top_left"
p.legend.click_policy = "hide"

hover = HoverTool(
    tooltips=[("Predicted", "@x{0.000}"), ("Observed", "@y{0.000}"), ("Count", "@count")],
    renderers=[combined_calibration_glyph] + list(city_glyphs.values())
)
p.add_tools(hover)

stats_div = Div(text="", width=1000)

# ---------------------------------------
# JS data and callback
# ---------------------------------------
js_data = {
    'calibration_data': [
        {
            'city': data['city'],
            'time_of_day_minutes': data['time_of_day_minutes'].tolist(),
            'probs': data['probs'].tolist(),
            'outcome': int(data['outcome']),
            'category': data['category']
        }
        for data in calibration_data
    ],
    'city_configs': {code: {'name': config['name'], 'color': config['color']} 
                    for code, config in CITY_CONFIGS.items()},
    'city_codes': list(CITY_CONFIGS.keys())
}

callback_js = CustomJS(
    args=dict(
        time_slider=time_slider,
        bins_select=bins_select,
        category_select=category_select,
        mode_select=mode_select,
        city_checkbox=city_checkbox,
        # Combined sources
        combined_calibration_source=combined_calibration_source,
        combined_individual_source=combined_individual_source,
        combined_base_rate_source=combined_base_rate_source,
        combined_band_source=combined_band_source,
        combined_calibration_glyph=combined_calibration_glyph,
        # City sources
        city_sources=city_sources,
        city_glyphs=city_glyphs,
        stats_div=stats_div,
        js_data=js_data,
        p=p
    ),
    code="""
    function minutes_to_time_label(minutes) {
        const m = Math.max(0, Math.round(minutes));
        const baseMin = 12 * 60;
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

    function get_active_cities() {
        return city_checkbox.active.map(i => js_data.city_codes[i]);
    }

    function filter_data_for_cities(cities, category_filter, current_time_minutes) {
        const preds = [], outs = [], city_labels = [];
        
        for (const data of js_data.calibration_data) {
            if (!cities.includes(data.city)) continue;
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
                city_labels.push(data.city);
            }
        }
        return {preds, outs, city_labels};
    }

    function calculate_calibration_bins(preds, outs, n_bins) {
        if (preds.length === 0) return {bin_pred: [], bin_obs: [], bin_counts: []};
        
        const edges = [];
        for (let i = 0; i <= n_bins; i++) edges.push(i / n_bins);

        const bin_pred = [], bin_obs = [], bin_counts = [];

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
                bin_pred.push(pbin.reduce((a,b)=>a+b,0)/pbin.length);
                bin_obs.push(obin.reduce((a,b)=>a+b,0)/obin.length);
                bin_counts.push(pbin.length);
            }
        }
        return {bin_pred, bin_obs, bin_counts};
    }

    function calculate_uncertainty_band(bin_pred, bin_counts) {
        const xs = [], lower = [], upper = [];
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
        return {xs, lower, upper};
    }

    function update_calibration() {
        const current_time_minutes = time_slider.value;
        const n_bins = parseInt(bins_select.value);
        const category_filter = category_select.value;
        const mode = mode_select.value;
        const active_cities = get_active_cities();

        // Hide all city-specific elements first
        for (const city_code of js_data.city_codes) {
            const sources = city_sources[city_code];
            const glyph = city_glyphs[city_code];
            
            sources.calibration.data = {x: [], y: [], size: [], count: [], color: [], line_color: []};
            sources.individual.data = {x: [], y: [], color: []};
            sources.base_rate.data = {x: [0,0], y0: [0,0], y1: [1,1]};
            sources.band.data = {xs: [], lower: [], upper: []};
            
            glyph.visible = false;
            // Note: Would need to store references to other renderers to hide them
        }

        if (mode === "combined") {
            // Show combined analysis
            combined_calibration_glyph.visible = true;
            
            const data = filter_data_for_cities(active_cities, category_filter, current_time_minutes);
            if (data.preds.length === 0) {
                combined_calibration_source.data = {x: [], y: [], size: [], count: [], color: [], line_color: []};
                combined_individual_source.data = {x: [], y: [], color: []};
                combined_base_rate_source.data = {x: [0,0], y0: [0,0], y1: [1,1]};
                combined_band_source.data = {xs: [], lower: [], upper: []};
                stats_div.text = "No data available for current selection";
                return;
            }

            const bins = calculate_calibration_bins(data.preds, data.outs, n_bins);
            
            // Point sizes and colors
            const sizes = bins.bin_counts.map(c => Math.max(8, Math.min(30, 8 + 3 * Math.log(c))));
            const colors = Array(bins.bin_pred.length).fill("#1f77b4");
            const lines = Array(bins.bin_pred.length).fill("#333333");

            combined_calibration_source.data = {
                x: bins.bin_pred, y: bins.bin_obs, size: sizes, 
                count: bins.bin_counts, color: colors, line_color: lines
            };

            combined_individual_source.data = {
                x: data.preds, y: data.outs, color: Array(data.preds.length).fill("black")
            };

            // Base rate
            const base_rate = data.outs.reduce((s,o)=>s+o,0) / data.outs.length;
            combined_base_rate_source.data = { x: [base_rate, base_rate], y0: [0,0], y1: [1,1] };

            // Uncertainty band
            const band = calculate_uncertainty_band(bins.bin_pred, bins.bin_counts);
            combined_band_source.data = band;

            // Stats
            const brier = data.preds.reduce((s,p,i)=> s + Math.pow(p - data.outs[i], 2), 0) / data.preds.length;
            const ref = data.outs.reduce((s,o)=> s + Math.pow(base_rate - o, 2), 0) / data.outs.length;
            const skill = ref > 0 ? 1 - (brier / ref) : NaN;

            const time_str = minutes_to_time_label(current_time_minutes);
            const city_names = active_cities.map(c => js_data.city_configs[c].name).join(", ");
            stats_div.text = `<b>Mode:</b> Combined Analysis | <b>Cities:</b> ${city_names}<br>` +
                           `<b>Time:</b> ${time_str} | <b>Markets:</b> ${data.preds.length} | ` +
                           `<b>Brier Score:</b> ${brier.toFixed(3)} | <b>Skill Score:</b> ${isNaN(skill) ? 'N/A' : skill.toFixed(3)}`;

        } else if (mode === "by_city") {
            // Show individual city analysis
            combined_calibration_glyph.visible = false;
            combined_calibration_source.data = {x: [], y: [], size: [], count: [], color: [], line_color: []};
            combined_individual_source.data = {x: [], y: [], color: []};
            combined_base_rate_source.data = {x: [0,0], y0: [0,0], y1: [1,1]};
            combined_band_source.data = {xs: [], lower: [], upper: []};

            let stats_text = `<b>Mode:</b> By City Analysis | <b>Time:</b> ${minutes_to_time_label(current_time_minutes)}<br>`;

            for (const city_code of active_cities) {
                const data = filter_data_for_cities([city_code], category_filter, current_time_minutes);
                const sources = city_sources[city_code];
                const glyph = city_glyphs[city_code];
                const color = js_data.city_configs[city_code].color;
                const name = js_data.city_configs[city_code].name;

                if (data.preds.length === 0) {
                    glyph.visible = false;
                    continue;
                }

                glyph.visible = true;
                
                const bins = calculate_calibration_bins(data.preds, data.outs, n_bins);
                const sizes = bins.bin_counts.map(c => Math.max(6, Math.min(25, 6 + 2 * Math.log(c))));
                
                sources.calibration.data = {
                    x: bins.bin_pred, y: bins.bin_obs, size: sizes,
                    count: bins.bin_counts, color: Array(bins.bin_pred.length).fill(color),
                    line_color: Array(bins.bin_pred.length).fill("#333333")
                };

                sources.individual.data = {
                    x: data.preds, y: data.outs, color: Array(data.preds.length).fill(color)
                };

                const base_rate = data.outs.reduce((s,o)=>s+o,0) / data.outs.length;
                sources.base_rate.data = { x: [base_rate, base_rate], y0: [0,0], y1: [1,1] };

                const band = calculate_uncertainty_band(bins.bin_pred, bins.bin_counts);
                sources.band.data = band;

                // Individual city stats
                const brier = data.preds.reduce((s,p,i)=> s + Math.pow(p - data.outs[i], 2), 0) / data.preds.length;
                const ref = data.outs.reduce((s,o)=> s + Math.pow(base_rate - o, 2), 0) / data.outs.length;
                const skill = ref > 0 ? 1 - (brier / ref) : NaN;

                stats_text += `<b>${name}:</b> ${data.preds.length} markets, Brier: ${brier.toFixed(3)}, Skill: ${isNaN(skill) ? 'N/A' : skill.toFixed(3)}<br>`;
            }

            stats_div.text = stats_text;

        } else if (mode === "comparison") {
            // Show comparison mode - similar to by_city but with different styling
            combined_calibration_glyph.visible = false;
            combined_calibration_source.data = {x: [], y: [], size: [], count: [], color: [], line_color: []};
            combined_individual_source.data = {x: [], y: [], color: []};
            combined_base_rate_source.data = {x: [0,0], y0: [0,0], y1: [1,1]};
            combined_band_source.data = {xs: [], lower: [], upper: []};

            let comparison_stats = [];
            
            for (const city_code of active_cities) {
                const data = filter_data_for_cities([city_code], category_filter, current_time_minutes);
                const sources = city_sources[city_code];
                const glyph = city_glyphs[city_code];
                const color = js_data.city_configs[city_code].color;
                const name = js_data.city_configs[city_code].name;

                if (data.preds.length === 0) {
                    glyph.visible = false;
                    continue;
                }

                glyph.visible = true;
                
                const bins = calculate_calibration_bins(data.preds, data.outs, n_bins);
                const sizes = bins.bin_counts.map(c => Math.max(6, Math.min(20, 6 + 2 * Math.log(c))));
                
                sources.calibration.data = {
                    x: bins.bin_pred, y: bins.bin_obs, size: sizes,
                    count: bins.bin_counts, color: Array(bins.bin_pred.length).fill(color),
                    line_color: Array(bins.bin_pred.length).fill("#333333")
                };

                // Don't show individual points in comparison mode to reduce clutter
                sources.individual.data = {x: [], y: [], color: []};

                const base_rate = data.outs.reduce((s,o)=>s+o,0) / data.outs.length;
                sources.base_rate.data = { x: [base_rate, base_rate], y0: [0,0], y1: [1,1] };

                // Lighter uncertainty bands in comparison mode
                const band = calculate_uncertainty_band(bins.bin_pred, bins.bin_counts);
                sources.band.data = band;

                const brier = data.preds.reduce((s,p,i)=> s + Math.pow(p - data.outs[i], 2), 0) / data.preds.length;
                const ref = data.outs.reduce((s,o)=> s + Math.pow(base_rate - o, 2), 0) / data.outs.length;
                const skill = ref > 0 ? 1 - (brier / ref) : NaN;

                comparison_stats.push({
                    name: name,
                    markets: data.preds.length,
                    base_rate: base_rate,
                    brier: brier,
                    skill: skill
                });
            }

            const time_str = minutes_to_time_label(current_time_minutes);
            let stats_text = `<b>Mode:</b> City Comparison | <b>Time:</b> ${time_str}<br>`;
            
            comparison_stats.sort((a, b) => a.brier - b.brier); // Sort by Brier score
            for (const stat of comparison_stats) {
                stats_text += `<b>${stat.name}:</b> ${stat.markets} mkts, Base: ${stat.base_rate.toFixed(3)}, ` +
                            `Brier: ${stat.brier.toFixed(3)}, Skill: ${isNaN(stat.skill) ? 'N/A' : stat.skill.toFixed(3)}<br>`;
            }

            stats_div.text = stats_text;
        }
    }

    // Initial update
    update_calibration();
    this.update_calibration = update_calibration;
    """
)

# Attach callbacks
time_slider.js_on_change('value', callback_js)
bins_select.js_on_change('value', callback_js)
category_select.js_on_change('value', callback_js)
mode_select.js_on_change('value', callback_js)
city_checkbox.js_on_change('active', callback_js)

# ---------------------------------------
# Layout and output
# ---------------------------------------
header = Div(
    text=f"<h2>Multi-City Temperature Market Calibration Analysis</h2>"
         f"<p>Interactive calibration across {len(calibration_data)} markets from {len(CITY_CONFIGS)} cities "
         f"over {len(set(d['event'] for d in calibration_data))} unique days "
         f"(window: prev day 12:00 PM → event day 9:00 PM ET).</p>"
         f"<p><b>Cities:</b> {', '.join(config['name'] for config in CITY_CONFIGS.values())}</p>"
         f"<p><b>Analysis Modes:</b> Combined (pool all cities), By City (separate analysis), Comparison (overlay cities)</p>",
    width=1000
)

city_controls = Div(text="<b>Select Cities:</b>", width=100)
controls_row1 = row(mode_select, bins_select, category_select, width=1000)
controls_row2 = row(city_controls, city_checkbox, width=1000)
controls_row3 = row(time_slider, width=1000)

layout = column(
    header, 
    controls_row1,
    controls_row2, 
    controls_row3, 
    p, 
    stats_div, 
    sizing_mode="scale_width"
)

output_file("kalshi_multi_city_temp_calibration.html", title="Multi-City Temperature Market Calibration Analysis")
print("Creating multi-city calibration plot...")
show(layout)

print("\n=== FILES CREATED ===")
print(f"✓ Cache file: {CACHE_FILE}")
if EXPORT_CSV:
    print(f"✓ CSV export: kalshi_multi_city_temp_data.csv")
else:
    print(f"- CSV export: Disabled (set EXPORT_CSV = True to enable)")
print(f"✓ HTML plot: kalshi_multi_city_temp_calibration.html")

print("\nConfiguration:")
print(f"- Date range: {FIRST_AVAILABLE_DATE.strftime('%Y-%m-%d')} to yesterday")
print(f"- Using {'ALL available data' if LOOKBACK_DAYS is None else f'lookback {LOOKBACK_DAYS} days'}")
print(f"- Total markets: {len(calibration_data)}")
print(f"- Unique days: {len(set(d['event'] for d in calibration_data))}")
print(f"- Cities analyzed: {len(CITY_CONFIGS)}")

if city_stats:
    print("\nPer-city breakdown:")
    for city_code, stats in city_stats.items():
        print(f"  {stats['name']}: {stats['markets']} markets, base rate {stats['base_rate']:.3f}")

print("\nTo force refresh data, set FORCE_REFRESH = True")
print("To export CSV data, set EXPORT_CSV = True (warning: can be large!)")
