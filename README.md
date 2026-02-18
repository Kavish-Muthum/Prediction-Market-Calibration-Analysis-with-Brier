# Multi-City Kalshi Temperature Market Calibration Analysis

---

## 🌦️ Project Website

> A deep dive into the methodology, data, and findings.

**[→ View the full write-up here](https://www.cs.utexas.edu/~kavish/blog/kalshi-weather-calibration.html)**

---

An interactive Bokeh web application for analyzing the calibration and forecasting skill of Kalshi temperature prediction markets across multiple cities. This tool downloads intraday market prices from seven major US cities, creates time-based snapshots, and provides comprehensive calibration analysis with uncertainty quantification and cross-city comparisons.

![GIF_Multi_City_Calibration.gif](https://github.com/Kavish-Muthum/Prediction-Market-Calibration-Analysis-with-Brier/blob/main/GIF_All_Calibration.gif?raw=true)

## Overview

This application evaluates how well Kalshi temperature markets are "calibrated" across multiple cities - whether events predicted with X% probability actually occur X% of the time. It provides:

- Interactive calibration plots with adjustable time windows across 7 cities
- Three analysis modes: Combined, By City, and Comparison
- Uncertainty bands showing statistical confidence intervals for each city
- Base rate visualization for reference comparison across regions
- Brier score metrics for quantitative skill assessment per city
- Time-series analysis from market open through event resolution

## Supported Cities

The analysis covers high temperature markets for seven major US cities:

| City Code | Market Ticker | City Name |
|-----------|---------------|-----------|
| NYC | KXHIGHNY | New York City |
| LAX | KXHIGHLAX | Los Angeles |
| PHIL | KXHIGHPHIL | Philadelphia |
| AUS | KXHIGHAUS | Austin |
| MIA | KXHIGHMIA | Miami |
| DEN | KXHIGHDEN | Denver |
| CHI | KXHRIGHCHI | Chicago |

## Features

### Core Analysis
- **Multi-City Calibration Curves**: Visual assessment of prediction accuracy across cities and probability bins
- **Comparative Analysis**: Three distinct modes for analyzing single cities vs. combined datasets
- **Brier Score Calculation**: Quantitative measure of forecasting skill per city
- **Skill Score Computation**: Performance relative to base rate predictions
- **Uncertainty Quantification**: Statistical confidence bands around perfect calibration

### Interactive Controls
- **Analysis Mode Selector**: Choose between Combined, By City, or Comparison analysis
- **City Selection**: Check/uncheck specific cities to include in analysis
- **Time Slider**: Analyze calibration at any point from previous day noon to event day 9 PM
- **Bin Selection**: Choose from 5, 10, 20, 50, or 100 probability bins
- **Category Filtering**: Filter by average market probability (very_low to very_high)

### Analysis Modes

1. **Combined Mode**: Pool all selected cities for maximum statistical power
   - Single calibration curve using all city data
   - Unified uncertainty bands and base rates
   - Most robust for overall market calibration assessment

2. **By City Mode**: Analyze each city separately
   - Individual calibration curves for each selected city
   - City-specific uncertainty bands and base rates
   - Ideal for identifying regional forecasting differences

3. **Comparison Mode**: Overlay cities on the same plot
   - Multiple calibration curves with city-specific colors
   - Side-by-side comparison of forecasting performance
   - Best for direct city-to-city skill comparison

### Data Management
- **Intelligent Multi-City Caching**: Automatic local caching with city-specific organization
- **Enhanced Export Options**: Summary CSV, full CSV, JSON, and Parquet formats
- **Robust API Handling**: Multiple fallback strategies for data retrieval across all cities

## Quick Start

### Prerequisites
```bash
pip install requests pandas numpy bokeh pickle5 zoneinfo
```

### Basic Usage
1. **Run the multi-city analysis**:
   ```bash
   python kalshi_multi_city_calibration.py
   ```

2. **Select your analysis preferences**:
   - Choose analysis mode (Combined/By City/Comparison)
   - Select cities to include in the analysis
   - Adjust time slider to focus on specific time windows

3. Open the generated HTML file in your browser to explore the interactive visualization

## Configuration Options

### Core Settings
```python
# Multi-City Configuration
CITY_CONFIGS = {
    "NYC": {"ticker": "KXHRIGHNY", "name": "New York City", "color": "#1f77b4"},
    "LAX": {"ticker": "KXHRIGHLAX", "name": "Los Angeles", "color": "#ff7f0e"},
    # ... additional cities
}

STATUS = "settled"                    # Only analyze settled markets

# Time Window (Eastern Time)
START_PREV_DAY_HOUR_ET = 12          # Start: Previous day noon
END_DAY_HOUR_ET = 21                 # End: Event day 9 PM

# Date Range (Updated for multi-city data availability)
FIRST_AVAILABLE_DATE = datetime(2025, 1, 25, tzinfo=LOCAL_TZ)  # 25JAN05
LOOKBACK_DAYS = None                 # None = all data, or specify days

# Performance & Storage
FORCE_REFRESH = False                # True to bypass cache
EXPORT_CSV = False                   # True to export full dataset
EXPORT_SUMMARY_CSV = True            # Export market-level summary (recommended)
```

### Analysis Parameters
```python
DEFAULT_BINS = 10                    # Default probability bins
BIN_OPTIONS = ["5", "10", "20", "50", "100"]  # Available bin counts
ANALYSIS_MODES = ["combined", "by_city", "comparison"]  # Available analysis modes
DEFAULT_MODE = "combined"            # Starting analysis mode
```

## Understanding the Multi-City Visualization

### Main Plot Elements

1. **Gray Dashed Line**: Perfect calibration (y = x)
   - Markets are perfectly calibrated if points fall on this line

2. **Colored Shaded Bands**: City-specific calibration uncertainty
   - 95% confidence intervals accounting for finite sample sizes
   - Colors match city-specific color scheme
   - Wider bands indicate fewer observations in that probability range

3. **Colored Dashed Vertical Lines**: City-specific base event rates
   - Overall frequency of YES outcomes for each city
   - Reference points for skill score calculations

4. **Colored Circles**: City-specific bin averages
   - Size indicates number of observations in each bin
   - Color coding matches each city's assigned color
   - Hover tooltips show predicted vs. observed frequencies

5. **Small Colored Dots**: Individual market outcomes (rug plot)
   - Color-coded by city
   - Y-axis: 0 for NO outcomes, 1 for YES outcomes
   - Opacity reduced in comparison mode to minimize clutter

### Mode-Specific Features

**Combined Mode**:
- Single blue calibration curve pooling all cities
- Unified statistics shown in status bar
- Maximum statistical power for overall assessment

**By City Mode**:
- Separate calibration curves for each selected city
- Individual city statistics displayed
- Best for identifying city-specific patterns

**Comparison Mode**:
- Overlaid calibration curves with city-specific colors
- Comparative statistics sorted by Brier score
- Individual points hidden to reduce visual clutter

### Statistical Metrics

- **Brier Score**: Lower is better (0 = perfect, 0.25 = random)
  - Measures the mean squared difference between predictions and outcomes
  - Formula: `BS = (1/N) × Σ(predicted_prob - outcome)²`

- **Skill Score**: Higher is better (1 = perfect, 0 = no skill)
  - Performance relative to always predicting the base rate
  - Formula: `SS = 1 - (Brier_Score / Brier_Reference)`

- **Base Rate**: City-specific frequency of high temperature events
  - Varies by city due to different climatic conditions
  - Important for contextualizing skill scores

## Technical Implementation

### Multi-City Data Pipeline

1. **Market Discovery**: 
   - Generates event tickers for all cities based on date patterns
   - Fetches market lists for each city's events via Kalshi API
   - Handles different data availability windows per city

2. **Price Data Collection**:
   - Downloads candlestick data for all cities in specified time windows
   - Handles multiple API endpoints and fallback intervals
   - Computes mid-prices from bid-ask spreads across all markets

3. **Outcome Resolution**:
   - Retrieves final settlement data for all city markets
   - Maps settlement prices to binary outcomes (YES/NO)
   - Maintains city attribution throughout processing

4. **Time Series Processing**:
   - Converts timestamps to Eastern Time (consistent across all cities)
   - Creates time-of-day indexing (minutes since window start)
   - Clips probabilities to avoid numerical issues

### Enhanced Caching Strategy

The application implements intelligent multi-city caching:

```python
# Multi-city cache file
CACHE_FILE = "kalshi_multi_city_temp_cache.pkl"

# Cache contains processed data for all cities
{
    'calibration_data': [
        {
            'city': 'NYC',
            'city_name': 'New York City',
            'event': 'KXRIGHNY-25JAN27',
            'market': 'KXRIGHNY-25JAN27-B65',
            'time_of_day_minutes': [0, 15, 30, ...],
            'probs': [0.45, 0.47, 0.52, ...],
            'outcome': 1,  # or 0
            'category': 'mid',
            'avg_prob': 0.48
        },
        # ... data for all cities
    ],
    'city_stats': {
        'NYC': {'name': 'New York City', 'markets': 150, 'yes': 75, 'no': 75, 'base_rate': 0.500},
        # ... stats for all cities
    }
}
```

### JavaScript Integration

The Bokeh application uses enhanced CustomJS callbacks for multi-city interactivity:
- City selection filtering
- Mode-specific visualization updates
- Dynamic binning calculations across cities
- Multi-city statistical metric updates
- City-specific uncertainty band computation

## Market Categories

Markets are automatically categorized by average probability across all cities:
- **very_low**: < 20%
- **low**: 20-40%  
- **mid**: 40-60%
- **high**: 60-80%
- **very_high**: > 80%

This enables analysis of calibration across different confidence levels and cities.

## Regional Climate Considerations

### Expected Base Rate Variations
Different cities will naturally have different base rates for high temperature events:
- **Miami**: Likely higher base rates due to consistently warm climate
- **Denver**: Potentially more variable due to altitude and continental climate
- **Los Angeles**: May show seasonal patterns due to Mediterranean climate
- **Chicago**: Likely shows strong seasonal variation

### Calibration Expectations
- Cities with more extreme climates may show better calibration due to clearer weather patterns
- Cities with more variable weather may present greater forecasting challenges
- Cross-city analysis helps identify whether calibration differences are due to market mechanics or climate predictability

## Important Notes

### API Rate Limits
- The Kalshi API has rate limits; processing all cities requires extended runtime
- Large date ranges across multiple cities may require significant time
- Intelligent caching minimizes repeated API calls during analysis

### Data Size Considerations
```python
# Multi-city exports can be very large
EXPORT_CSV = False           # Enable with caution
EXPORT_SUMMARY_CSV = True    # Recommended for initial analysis
# Estimated sizes displayed before export
```

### Time Zone Handling
- All times converted to Eastern Time for consistency across cities
- Market windows assume Eastern Time trading hours for all cities
- Timestamps properly handled across daylight saving transitions

## Interpretation Guidelines

### Multi-City Calibration Assessment

**Good Overall Calibration**:
- Points cluster near diagonal across most cities
- Uncertainty bands contain the diagonal consistently
- Similar calibration patterns across different climates

**Regional Calibration Differences**:
- Systematic city-specific deviations from diagonal
- Varying uncertainty band widths (sample size effects)
- Base rate differences reflecting local climate patterns

**Skill Assessment Across Cities**:
- Compare Brier scores across cities in Comparison mode
- Look for consistent skill patterns vs. city-specific anomalies
- Consider base rate differences when interpreting skill scores

### Mode-Specific Analysis

**Combined Mode Analysis**:
- Use for overall market calibration assessment
- Maximum statistical power for detecting systematic biases
- Best for general conclusions about Kalshi temperature markets

**By City Mode Analysis**:
- Identify cities with particularly good/poor calibration
- Understand regional forecasting challenges
- Detect city-specific market behavior patterns

**Comparison Mode Analysis**:
- Direct skill comparison across cities
- Identify consistently well-calibrated vs. problematic markets
- Understand relative forecasting difficulty by region

## Customization Options

### Adding New Cities
```python
# Add new cities to the configuration
CITY_CONFIGS["SEATTLE"] = {
    "ticker": "KXRIGHSEA", 
    "name": "Seattle", 
    "color": "#2ca02c"
}
```

### Adjusting Time Windows
```python
# Analyze different time periods
START_PREV_DAY_HOUR_ET = 8   # Start at 8 AM previous day
END_DAY_HOUR_ET = 23         # End at 11 PM event day
```

### Custom Analysis Modes
```python
# Add new analysis modes in the JavaScript callback
ANALYSIS_MODES = ["combined", "by_city", "comparison", "regional"]
```

## Output Files

Running the multi-city application creates several files:

1. **`kalshi_multi_city_temp_cache.pkl`**: Processed data cache for all cities
2. **`kalshi_multi_city_temp_calibration.html`**: Interactive multi-city visualization
3. **`kalshi_multi_city_temp_summary.csv`**: Market-level summary data (recommended)
4. **`kalshi_multi_city_temp_data.csv`**: Full time series data (if enabled)
5. **`kalshi_multi_city_temp_data.json`**: Structured JSON export (if enabled)

## Troubleshooting

### Common Multi-City Issues

**"No calibration data loaded"**
- Check internet connection for API access across all cities
- Verify city tickers exist and have settled markets
- Ensure date range includes available data for multiple cities

**Uneven city data**
- Some cities may have different market availability windows
- Use city selection controls to focus on cities with sufficient data
- Check city-specific statistics in the output

**Large memory usage**
- Reduce LOOKBACK_DAYS for shorter analysis periods
- Disable full CSV export for memory-intensive runs
- Use fewer cities for initial testing

**Slow performance**
- Enable caching (FORCE_REFRESH = False)
- Process cities sequentially (handled automatically)
- Use smaller date ranges for initial testing

### Debug Configuration
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Force fresh data download
FORCE_REFRESH = True

# Reduce scope for testing
LOOKBACK_DAYS = 7
CITY_CONFIGS = {"NYC": CITY_CONFIGS["NYC"]}  # Test with single city
```

## Sample Output Statistics

Expected output for multi-city analysis:
```
=== COMBINED SUMMARY ===
Total markets: 1,247
Total outcomes: 623 YES, 624 NO
Overall base rate: 0.499
Unique days: 178

Per-city breakdown:
  New York City: 187 markets, base rate 0.487
  Los Angeles: 169 markets, base rate 0.521
  Philadelphia: 182 markets, base rate 0.478
  Austin: 174 markets, base rate 0.534
  Miami: 156 markets, base rate 0.628
  Denver: 203 markets, base rate 0.425
  Chicago: 176 markets, base rate 0.460
```

## References

- **Calibration Theory**: [Calibration and Skill of the Kalshi Prediction Markets](https://www.cwdatasolutions.com/post/calibration-and-skill-of-the-kalshi-prediction-markets)
- **Brier Score**: [Verification of Forecasts Expressed in Terms of Probability](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
- **Kalshi API**: [Official Documentation](https://kalshi-public-docs.s3.amazonaws.com/KalshiAPI.html)

---

*Built with Python, Pandas, and Bokeh for interactive multi-city market analysis*
