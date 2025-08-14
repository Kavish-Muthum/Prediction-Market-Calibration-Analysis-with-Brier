# Kalshi Market Calibration Analysis

An interactive Bokeh web application for analyzing the calibration and forecasting skill of Kalshi prediction markets. This tool downloads intraday market prices, creates time-based snapshots, and provides comprehensive calibration analysis with uncertainty quantification.

![Kalshi Calibration Analysis Example](https://github.com/Kavish-Muthum/Prediction-Market-Calibration-Analysis-with-Brier/blob/main/GIF_Calibration.gif?raw=true)

## Overview

This application evaluates how well Kalshi markets are "calibrated" - whether events predicted with X% probability actually occur X% of the time. It provides:

- Interactive calibration plots with adjustable time windows
- Uncertainty bands showing statistical confidence intervals
- Base rate visualization for reference comparison
- Brier score metrics for quantitative skill assessment
- Time-series analysis from market open through event resolution

## Features

### Core Analysis
- Calibration Curves: Visual assessment of prediction accuracy across probability bins
- Brier Score Calculation: Quantitative measure of forecasting skill
- Skill Score Computation: Performance relative to base rate predictions
- Uncertainty Quantification: Statistical confidence bands around perfect calibration

### Interactive Controls
- Time Slider: Analyze calibration at any point from previous day noon to event day 9 PM
- Bin Selection: Choose from 5, 10, 20, 50, or 100 probability bins
- Category Filtering: Filter by average market probability (very_low to very_high)

### Data Management
- Intelligent Caching: Automatic local caching with configurable refresh
- CSV Export: Optional full dataset export for external analysis
- Robust API Handling: Multiple fallback strategies for data retrieval

## Quick Start

### Prerequisites
```bash
pip install requests pandas numpy bokeh pickle5 zoneinfo
```

### Basic Usage
1. **Configure the analysis**:
   ```python
   SERIES_TICKER = "KXHIGHNY"  # Change to your target market series
   LOOKBACK_DAYS = 30          # Or None for all available data
   ```

2. **Run the analysis**:
   ```bash
   python kalshi_calibration.py
   ```

3. Open the generated HTML file in your browser to explore the interactive visualization

## Configuration Options

### Core Settings
```python
# Market Configuration
SERIES_TICKER = "KXHIGHNY"           # Target market series
STATUS = "settled"                    # Only analyze settled markets

# Time Window (Eastern Time)
START_PREV_DAY_HOUR_ET = 12          # Start: Previous day noon
END_DAY_HOUR_ET = 21                 # End: Event day 9 PM

# Date Range
FIRST_AVAILABLE_DATE = datetime(2024, 10, 25, tzinfo=LOCAL_TZ)
LOOKBACK_DAYS = None                 # None = all data, or specify days

# Performance & Storage
FORCE_REFRESH = False                # True to bypass cache
EXPORT_CSV = False                   # True to export full dataset
```

### Analysis Parameters
```python
DEFAULT_BINS = 10                    # Default probability bins
BIN_OPTIONS = ["5", "10", "20", "50", "100"]  # Available bin counts
```

## Understanding the Visualization

### Main Plot Elements

1. Gray Dashed Line: Perfect calibration (y = x)
   - Markets are perfectly calibrated if points fall on this line

2. Blue Shaded Band: Calibration uncertainty
   - 95% confidence interval accounting for finite sample sizes
   - Wider bands indicate fewer observations in that probability range

3. Blue Dashed Vertical Line: Base event rate
   - Overall frequency of YES outcomes in the selected data
   - Reference point for skill score calculations

4. Colored Circles: Bin averages
   - Size indicates number of observations in each bin
   - Color coding matches Bokeh's Category10 palette

5. Small Gray Dots: Individual market outcomes
   - "Rug plot" showing raw data points
   - Y-axis: 0 for NO outcomes, 1 for YES outcomes

### Statistical Metrics

- Brier Score: Lower is better (0 = perfect, 0.25 = random)
  - Measures the mean squared difference between predictions and outcomes
  - Formula: `BS = (1/N) × Σ(predicted_prob - outcome)²`

- Skill Score: Higher is better (1 = perfect, 0 = no skill)
  - Performance relative to always predicting the base rate
  - Formula: `SS = 1 - (Brier_Score / Brier_Reference)`

## Technical Implementation

### Data Pipeline

1. Market Discovery: 
   - Generates event tickers based on date patterns
   - Fetches market lists for each event via Kalshi API

2. Price Data Collection:
   - Downloads candlestick data for specified time windows
   - Handles multiple API endpoints and fallback intervals
   - Computes mid-prices from bid-ask spreads

3. Outcome Resolution:
   - Retrieves final settlement data
   - Maps settlement prices to binary outcomes (YES/NO)

4. Time Series Processing:
   - Converts timestamps to Eastern Time
   - Creates time-of-day indexing (minutes since window start)
   - Clips probabilities to avoid numerical issues

### Caching Strategy

The application implements intelligent caching to minimize API calls:

```python
# Cache file naming
CACHE_FILE = f"kalshi_{SERIES_TICKER}_cache.pkl"

# Cache contains processed data for all markets
{
    'event': 'KXHIGHNY-24NOV15',
    'market': 'KXHIGHNY-24NOV15-B75',
    'time_of_day_minutes': [0, 15, 30, ...],
    'probs': [0.45, 0.47, 0.52, ...],
    'outcome': 1,  # or 0
    'category': 'mid',
    'avg_prob': 0.48
}
```

### JavaScript Integration

The Bokeh application uses CustomJS callbacks for real-time interactivity:
- Time-based data filtering
- Dynamic binning calculations  
- Statistical metric updates
- Uncertainty band computation

## Market Categories

Markets are automatically categorized by average probability:
- very_low: < 20%
- low: 20-40%  
- mid: 40-60%
- high: 60-80%
- very_high: > 80%

This enables analysis of calibration across different confidence levels.

## Important Notes

### API Rate Limits
- The Kalshi API has rate limits; the code includes timeout handling
- Large date ranges may require extended runtime
- Use caching to avoid repeated API calls during development

### Data Size Considerations
```python
# CSV exports can be very large
EXPORT_CSV = False  # Enable with caution
# Estimated size displayed before export
```

### Time Zone Handling
- All times are converted to Eastern Time for consistency
- Market windows assume Eastern Time trading hours
- Timestamps are properly handled across daylight saving transitions

## Interpretation Guidelines

### Good Calibration Indicators
- Points cluster near the diagonal line
- Uncertainty bands contain the diagonal
- Consistent performance across probability ranges

### Poor Calibration Signs
- Systematic deviations from diagonal (overconfidence/underconfidence)
- Points consistently outside uncertainty bands
- Large variations between probability bins

### Skill Assessment
- Brier Score < Base Rate Variance: Indicates forecasting skill
- Positive Skill Score: Better than base rate predictions
- Skill Score > 0.1: Generally considered meaningful improvement

## Customization Options

### Extending to New Markets
```python
# Change the series ticker to analyze different markets
SERIES_TICKER = "KXHIGHAUS"          # Austin Highest temperature
```

### Adjusting Time Windows
```python
# Analyze different time periods
START_PREV_DAY_HOUR_ET = 8   # Start at 8 AM previous day
END_DAY_HOUR_ET = 23         # End at 11 PM event day
```

### Custom Probability Categories
```python
# Modify category thresholds in the binning logic
if   avg_prob < 0.1: category = "extremely_low"
elif avg_prob < 0.3: category = "low"
# ... etc
```

## 📁 Output Files

Running the application creates several files:

1. **`kalshi_{SERIES_TICKER}_cache.pkl`**: Processed data cache
2. **`kalshi_{SERIES_TICKER}_calibration.html`**: Interactive visualization
3. **`kalshi_{SERIES_TICKER}_data.csv`**: Raw data export (if enabled)

## Troubleshooting

### Common Issues

**"No calibration data loaded"**
- Check internet connection for API access
- Verify SERIES_TICKER exists and has settled markets
- Ensure date range includes available data

**Large memory usage**
- Reduce LOOKBACK_DAYS for shorter analysis periods
- Disable CSV export for memory-intensive runs

**Slow performance**
- Enable caching (FORCE_REFRESH = False)
- Use smaller date ranges for initial testing
- Check API response times

### Debug Configuration
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Force fresh data download
FORCE_REFRESH = True

# Reduce scope for testing
LOOKBACK_DAYS = 7
```

## References

- **Calibration Theory**: [Calibration and Skill of the Kalshi Prediction Markets](https://www.cwdatasolutions.com/post/calibration-and-skill-of-the-kalshi-prediction-markets)
- **Brier Score**: [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
- **Kalshi API**: [Official Documentation](https://kalshi-public-docs.s3.amazonaws.com/KalshiAPI.html)

---

*Built with Python, Pandas, and Bokeh for interactive market analysis*
