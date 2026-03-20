# Yahoo Finance MCP Server

A comprehensive MCP (Model Context Protocol) server that provides financial data and analysis tools using Yahoo Finance. This server enables AI assistants to access real-time stock data, perform technical analysis, manage portfolios, and much more.

Yahoo Finance provides a comprehensive suite of financial data, including real-time and historical stock quotes, market indices, currencies, and cryptocurrencies from over 100 global exchanges. It covers fundamental data like 5 years of financial statements (via Morningstar), analyst recommendations, EPS/revenue estimates (via S&P Global), and market news. 


* Asset Classes Covered: Equities, World Indices, ETFs, Currencies/FX, Cryptocurrencies
:
* Market Data: Real-time (with some delays) and historical price, dividend, and split data for stocks, ETFs, cryptocurrencies, foreign exchange rates.

* Fundamental Data: Balance sheets, income statements, cash flow, valuation ratios, market cap, and shares outstanding.

* Analytics & Research: Analyst recommendations, price targets, earnings estimates, and insider transactions.

* Coverage: Extensive global coverage, including major indices (Dow Jones, Nasdaq, S&P) and various international exchanges.

## Installation

```bash
# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.11
- mcp[cli] >= 1.6.0
- yfinance >= 0.2.62
- numpy >= 1.24.0

## Running the Server

```bash
python server.py
```

The server runs using stdio transport by default.

---

## Available Tools

### Core Data Tools

#### `get_historical_stock_prices`
Get historical OHLCV (Open, High, Low, Close, Volume) data for a stock.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol (e.g., "AAPL") |
| `period` | str | "1mo" | Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max |
| `interval` | str | "1d" | Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo |

**Example Response:**
```json
[
  {"Date": "2024-01-15", "Open": 150.25, "High": 152.30, "Low": 149.80, "Close": 151.50, "Volume": 45000000}
]
```

---

#### `get_stock_info`
Get comprehensive stock information including price, company info, financial metrics, and more.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** JSON object with stock price, trading info, company information, financial metrics, earnings, margins, dividends, balance sheet data, ownership info, analyst coverage, and risk metrics.

---

#### `get_yahoo_finance_news`
Get recent news articles for a stock.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** List of news articles with title, summary, description, and URL.

---

#### `get_stock_actions`
Get dividend and stock split history.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

---

#### `get_financial_statement`
Get financial statements (income statement, balance sheet, or cash flow).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `financial_type` | str | required | One of: `income_stmt`, `quarterly_income_stmt`, `balance_sheet`, `quarterly_balance_sheet`, `cashflow`, `quarterly_cashflow` |

---

#### `get_holder_info`
Get holder information for a stock.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `holder_type` | str | required | One of: `major_holders`, `institutional_holders`, `mutualfund_holders`, `insider_transactions`, `insider_purchases`, `insider_roster_holders` |

---

#### `get_option_expiration_dates`
Get available options expiration dates for a stock.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

---

#### `get_option_chain`
Get options chain data (calls or puts) for a specific expiration date.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `expiration_date` | str | required | Expiration date (YYYY-MM-DD format) |
| `option_type` | str | required | "calls" or "puts" |

---

#### `get_recommendations`
Get analyst recommendations and upgrades/downgrades.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `recommendation_type` | str | required | "recommendations" or "upgrades_downgrades" |
| `months_back` | int | 12 | Number of months back for upgrades/downgrades |

---

### Technical Analysis Tools

#### `get_moving_averages`
Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `periods` | list[int] | [20, 50, 200] | List of periods to calculate |
| `ma_type` | str | "both" | "sma", "ema", or "both" |

**Example Response:**
```json
{
  "ticker": "AAPL",
  "currentPrice": 185.50,
  "movingAverages": {
    "20day": {"SMA": 182.30, "EMA": 183.10, "priceVsSMA": "1.75%"},
    "50day": {"SMA": 178.50, "EMA": 179.20, "priceVsSMA": "3.92%"}
  }
}
```

---

#### `get_rsi`
Calculate the Relative Strength Index (RSI).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `period` | int | 14 | RSI calculation period |

**Returns:** Current RSI value, signal (Overbought/Oversold/Neutral), and recent RSI values.

---

#### `get_macd`
Calculate the MACD (Moving Average Convergence Divergence) indicator.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `fast_period` | int | 12 | Fast EMA period |
| `slow_period` | int | 26 | Slow EMA period |
| `signal_period` | int | 9 | Signal line period |

**Returns:** MACD line, signal line, histogram, and signal (Bullish/Bearish/Crossover).

---

#### `get_bollinger_bands`
Calculate Bollinger Bands.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `period` | int | 20 | SMA period for middle band |
| `std_dev` | int | 2 | Number of standard deviations |

**Returns:** Upper band, middle band, lower band, %B indicator, bandwidth, and signal.

---

#### `get_support_resistance_levels`
Calculate support and resistance levels using pivot points.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `period` | str | "6mo" | Historical period to analyze |

**Returns:** Pivot point, R1/R2 resistance levels, S1/S2 support levels, and 52-week high/low.

---

#### `get_technical_summary`
Get a comprehensive technical analysis summary with multiple indicators.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** Moving averages (SMA/EMA), oscillators (RSI, MACD), Bollinger Bands, volume analysis, combined signals, and overall signal (Bullish/Bearish/Neutral).

---

### Portfolio Management Tools

#### `calculate_portfolio_value`
Calculate the current value of a portfolio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holdings` | dict | required | Dictionary of ticker:quantity pairs, e.g., `{"AAPL": 10, "GOOGL": 5}` |

**Returns:** Total portfolio value, individual position values, and portfolio weights.

---

#### `get_portfolio_allocation`
Get sector and industry breakdown of a portfolio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holdings` | dict | required | Dictionary of ticker:quantity pairs |

**Returns:** Sector allocation, industry allocation, diversification score, and top sector.

---

#### `calculate_portfolio_returns`
Calculate portfolio returns over a specified period.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holdings` | dict | required | Dictionary of ticker:quantity pairs |
| `period` | str | "1mo" | Period: "1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd" |

**Returns:** Start/end values, total return percentage, gain/loss, and per-position returns.

---

#### `get_portfolio_risk_metrics`
Calculate portfolio risk metrics including volatility, beta, and Sharpe ratio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holdings` | dict | required | Dictionary of ticker:quantity pairs |
| `benchmark` | str | "SPY" | Benchmark ticker for comparison |

**Returns:** Annualized volatility, portfolio beta, Sharpe ratio, expected annual return, Value at Risk (95%), and individual stock risk metrics.

---

#### `get_correlation_matrix`
Calculate correlation matrix for multiple stocks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `period` | str | "1y" | Historical period for correlation calculation |

**Returns:** Correlation matrix, highly correlated pairs, average correlation, and diversification insight.

---

### Earnings & Events Tools

#### `get_earnings_calendar`
Get upcoming and recent earnings dates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

---

#### `get_earnings_history`
Get historical earnings data (EPS actual vs estimates).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** Earnings history, quarterly earnings, and annual earnings data.

---

#### `get_analyst_estimates`
Get analyst earnings and revenue estimates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** Earnings estimates, revenue estimates, EPS trend, and growth estimates.

---

#### `get_upcoming_events`
Get upcoming events for multiple stocks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |

**Returns:** Earnings dates, ex-dividend dates, and other upcoming events.

---

### Peer Comparison & Industry Analysis

#### `get_sector_performance`
Get performance metrics for major market sectors using sector ETFs.

**No parameters required.**

**Returns:** Performance data for Technology (XLK), Healthcare (XLV), Financial (XLF), and other major sectors including YTD return, month return, and 52-week range.

---

#### `get_industry_peers`
Find peer companies in the same industry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `limit` | int | 10 | Maximum number of peers to return |

**Returns:** List of peer companies with market cap and P/E ratio.

---

#### `compare_to_peers`
Compare a stock's key metrics to its peers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Main stock to compare |
| `peer_tickers` | list[str] | required | List of peer tickers |
| `metrics` | list[str] | ["pe", "pb", "ps", "margin", "growth", "dividend"] | Metrics to compare |

---

### Valuation Tools

#### `get_valuation_ratios`
Get comprehensive valuation ratios.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** Price multiples (P/E, P/B, P/S, PEG), enterprise value metrics (EV/Revenue, EV/EBITDA), and profitability ratios (ROE, ROA, margins).

---

#### `get_dcf_inputs`
Get inputs needed for DCF (Discounted Cash Flow) valuation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |

**Returns:** Free cash flow history, growth rates, risk metrics, and suggested WACC/terminal growth rate ranges.

---

#### `calculate_intrinsic_value`
Calculate estimated intrinsic value using a simplified DCF model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `growth_rate` | float | 0.10 | Expected annual growth rate (e.g., 0.10 = 10%) |
| `discount_rate` | float | 0.10 | Discount rate / WACC |
| `terminal_growth` | float | 0.025 | Terminal growth rate |

**Returns:** Projected FCF, present value calculations, intrinsic value per share, and buy/sell recommendation.

---

### Currency & Crypto Tools

#### `get_exchange_rate`
Get currency exchange rate between two currencies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_currency` | str | required | Source currency (e.g., "USD") |
| `to_currency` | str | required | Target currency (e.g., "EUR") |

**Returns:** Current rate, daily change, historical rates (week/month ago), and conversion examples.

---

#### `get_crypto_price`
Get cryptocurrency price and information.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | required | Crypto symbol with currency pair (e.g., "BTC-USD", "ETH-USD") |

**Returns:** Current price, day's OHLCV, month stats, and returns (day/week/month).

---

#### `get_crypto_comparison`
Compare multiple cryptocurrencies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbols` | list[str] | required | List of crypto symbols (e.g., ["BTC-USD", "ETH-USD"]) |

**Returns:** Comparison of prices, market caps, year returns, and volatility.

---

### Index & ETF Tools

#### `get_major_indices`
Get current values and performance of major market indices.

**No parameters required.**

**Returns:** S&P 500, Dow Jones, NASDAQ, Russell 2000, VIX, 10Y Treasury data with current values and daily changes.

---

#### `get_etf_holdings`
Get ETF information and top holdings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `etf_ticker` | str | required | ETF ticker symbol (e.g., "SPY", "QQQ") |

**Returns:** ETF name, category, total assets, expense ratio, and return metrics.

---

#### `compare_to_benchmark`
Compare a stock's performance to a benchmark.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker to compare |
| `benchmark` | str | "SPY" | Benchmark ticker |
| `period` | str | "1y" | Comparison period |

**Returns:** Stock vs benchmark returns, volatility comparison, beta, alpha, and outperformance analysis.

---

### Alert & Watchlist Features

#### `check_price_targets`
Check if stocks have reached specified price targets.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `watchlist` | dict | required | Dictionary of ticker:target_price pairs, e.g., `{"AAPL": 200, "GOOGL": 150}` |

**Returns:** Triggered and pending alerts with distance to target.

---

#### `get_52_week_alerts`
Get alerts for stocks near their 52-week highs or lows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `threshold` | float | 5 | Percentage threshold from high/low |

**Returns:** Stocks near 52-week highs and lows with distance percentages.

---

#### `get_unusual_volume`
Detect stocks with unusual trading volume.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `threshold` | float | 2.0 | Volume ratio threshold (e.g., 2.0 = 2x average) |

**Returns:** Stocks with unusual volume, volume ratio, and price change.

---

#### `get_significant_movers`
Get stocks with significant price moves.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `threshold` | float | 3.0 | Minimum percentage move |

**Returns:** Top gainers and losers with price changes.

---

### Historical Analysis Tools

#### `calculate_returns`
Calculate returns over various periods.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `periods` | list[str] | ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"] | Periods to calculate |

**Returns:** Returns for each specified period.

---

#### `get_volatility_analysis`
Get comprehensive volatility analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `period` | str | "1y" | Historical period to analyze |

**Returns:** Daily/annualized volatility, rolling volatility, ATR, price range, beta, and volatility rating.

---

#### `get_max_drawdown`
Calculate maximum drawdown for a stock.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ticker` | str | required | Stock ticker symbol |
| `period` | str | "1y" | Historical period to analyze |

**Returns:** Maximum drawdown percentage, peak/trough dates and values, recovery days, and current drawdown.

---

#### `compare_performance`
Compare performance of multiple stocks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `period` | str | "1y" | Comparison period |

**Returns:** Ranked performance comparison with returns, volatility, Sharpe ratio, and max drawdown for each stock.

---

### Screening Tools

#### `filter_stocks_by_price`
Filter stocks by price range.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols to filter |
| `min_price` | float | 0 | Minimum price |
| `max_price` | float | None | Maximum price (None = no limit) |

---

#### `filter_stocks_by_market_cap`
Filter stocks by market capitalization category.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `market_cap_category` | str | required | "mega" (>$200B), "large" ($10B-$200B), "mid" ($2B-$10B), "small" ($300M-$2B), "micro" (<$300M) |

---

#### `filter_stocks_by_pe_ratio`
Filter stocks by P/E ratio range.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `min_pe` | float | 0 | Minimum P/E ratio |
| `max_pe` | float | None | Maximum P/E ratio |
| `use_forward_pe` | bool | False | Use forward P/E instead of trailing |

---

#### `filter_stocks_by_dividend_yield`
Filter stocks by dividend yield category.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `dividend_category` | str | required | "high_yield" (>4%), "medium_yield" (2-4%), "low_yield" (0-2%), "no_dividend" (0%) |

---

#### `screen_stocks`
Screen stocks using multiple criteria.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |
| `min_price` | float | None | Minimum price |
| `max_price` | float | None | Maximum price |
| `min_market_cap` | float | None | Minimum market cap (in billions) |
| `max_market_cap` | float | None | Maximum market cap (in billions) |
| `min_pe` | float | None | Minimum P/E ratio |
| `max_pe` | float | None | Maximum P/E ratio |
| `min_dividend_yield` | float | None | Minimum dividend yield (as percentage) |
| `sector` | str | None | Filter by sector name |

---

#### `get_stocks_summary`
Get a quick summary comparison of multiple stocks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tickers` | list[str] | required | List of ticker symbols |

**Returns:** Key metrics for each stock including price, market cap, P/E ratios, dividend yield, 52-week range, moving averages, beta, margins, and revenue growth.

---

## Usage Examples

### MCP Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "yahoo-finance": {
      "command": "python",
      "args": ["path/to/server.py"],
      "transport": "stdio"
    }
  }
}
```

### Example Queries

**Get Apple stock info:**
```
get_stock_info(ticker="AAPL")
```

**Technical analysis:**
```
get_technical_summary(ticker="MSFT")
```

**Portfolio analysis:**
```
calculate_portfolio_value(holdings={"AAPL": 50, "GOOGL": 20, "MSFT": 30})
get_portfolio_risk_metrics(holdings={"AAPL": 50, "GOOGL": 20, "MSFT": 30})
```

**Screen for value stocks:**
```
screen_stocks(
    tickers=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    max_pe=25,
    min_dividend_yield=1
)
```

---

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
