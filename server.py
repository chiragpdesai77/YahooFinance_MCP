import json
from enum import Enum
from datetime import datetime
import numpy as np

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP


# Cache helper for stock data
_cache = {}
_cache_ttl = 300  # 5 minutes


def get_cached_ticker(ticker: str) -> yf.Ticker:
    """Get a ticker with basic caching"""
    cache_key = f"ticker_{ticker}"
    now = datetime.now().timestamp()
    
    if cache_key in _cache:
        cached_data, timestamp = _cache[cache_key]
        if now - timestamp < _cache_ttl:
            return cached_data
    
    company = yf.Ticker(ticker)
    _cache[cache_key] = (company, now)
    return company


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


class MarketCapCategory(str, Enum):
    mega = "mega"  # > 200B
    large = "large"  # 10B - 200B
    mid = "mid"  # 2B - 10B
    small = "small"  # 300M - 2B
    micro = "micro"  # < 300M


class DividendFilter(str, Enum):
    high_yield = "high_yield"  # > 4%
    medium_yield = "medium_yield"  # 2% - 4%
    low_yield = "low_yield"  # 0% - 2%
    no_dividend = "no_dividend"  # 0%


# Initialize FastMCP server
yfinance_server = FastMCP(
    "yfinance",
    instructions="""
# Yahoo Finance MCP Server

This server provides comprehensive financial data and analysis tools using Yahoo Finance.

## Core Data Tools:
- get_historical_stock_prices: Get historical OHLCV data for a ticker
- get_stock_info: Get comprehensive stock information
- get_yahoo_finance_news: Get news for a ticker
- get_stock_actions: Get dividends and stock splits
- get_financial_statement: Get income statement, balance sheet, or cashflow (annual/quarterly)
- get_holder_info: Get major holders, institutional holdings, insider transactions
- get_option_expiration_dates: Get available options expiration dates
- get_option_chain: Get calls/puts option chain data
- get_recommendations: Get analyst recommendations and upgrades/downgrades

## Technical Analysis Tools:
- get_moving_averages: Calculate SMA and EMA for various periods
- get_rsi: Calculate Relative Strength Index
- get_macd: Calculate MACD indicator
- get_bollinger_bands: Calculate Bollinger Bands
- get_support_resistance_levels: Calculate support/resistance and pivot points
- get_technical_summary: Comprehensive technical analysis with multiple indicators

## Portfolio Management Tools:
- calculate_portfolio_value: Calculate current portfolio value and positions
- get_portfolio_allocation: Get sector/industry breakdown of portfolio
- calculate_portfolio_returns: Calculate portfolio returns over a period
- get_portfolio_risk_metrics: Calculate volatility, beta, Sharpe ratio, VaR
- get_correlation_matrix: Calculate correlation between multiple stocks

## Earnings & Events Tools:
- get_earnings_calendar: Get upcoming and recent earnings dates
- get_earnings_history: Get historical EPS data and estimates
- get_analyst_estimates: Get analyst earnings and revenue estimates
- get_upcoming_events: Get upcoming events for multiple stocks

## Peer Comparison & Industry Analysis:
- get_sector_performance: Get performance of major market sectors
- get_industry_peers: Find peer companies in the same industry
- compare_to_peers: Compare key metrics across peer group

## Valuation Tools:
- get_valuation_ratios: Get comprehensive valuation ratios (PE, PB, PS, EV/EBITDA)
- get_dcf_inputs: Get inputs for DCF valuation
- calculate_intrinsic_value: Calculate intrinsic value using simplified DCF

## Currency & Crypto:
- get_exchange_rate: Get forex exchange rates
- get_crypto_price: Get cryptocurrency prices and info
- get_crypto_comparison: Compare multiple cryptocurrencies

## Index & ETF Tools:
- get_major_indices: Get major market indices values
- get_etf_holdings: Get ETF information and top holdings
- compare_to_benchmark: Compare stock performance to benchmark

## Alert & Watchlist Features:
- check_price_targets: Check if stocks hit target prices
- get_52_week_alerts: Alert for stocks near 52-week highs/lows
- get_unusual_volume: Detect unusual trading volume
- get_significant_movers: Find stocks with significant price moves

## Historical Analysis:
- calculate_returns: Calculate returns over various periods
- get_volatility_analysis: Analyze stock volatility
- get_max_drawdown: Calculate maximum drawdown
- compare_performance: Compare performance of multiple stocks

## Screening Tools:
- filter_stocks_by_price: Filter stocks by price range
- filter_stocks_by_market_cap: Filter by market cap category
- filter_stocks_by_pe_ratio: Filter by P/E ratio range
- filter_stocks_by_dividend_yield: Filter by dividend yield category
- screen_stocks: Screen stocks with multiple criteria
- get_stocks_summary: Quick comparison of multiple stocks
""",
)


@yfinance_server.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        Default is "1mo"
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        Default is "1d"
""",
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    """Get historical stock prices for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
            Default is "1mo"
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting historical stock prices for {ticker}: {e}")
        return f"Error: getting historical stock prices for {ticker}: {e}"

    # If the company is found, get the historical data
    hist_data = company.history(period=period, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    hist_data = hist_data.to_json(orient="records", date_format="iso")
    return hist_data


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
)
async def get_stock_info(ticker: str) -> str:
    """Get stock information for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting stock information for {ticker}: {e}")
        return f"Error: getting stock information for {ticker}: {e}"
    info = company.info
    return json.dumps(info)


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    """Get news for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    # If the company is found, get the news
    try:
        news = company.news
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    news_list = []
    for news in company.news:
        if news.get("content", {}).get("contentType", "") == "STORY":
            title = news.get("content", {}).get("title", "")
            summary = news.get("content", {}).get("summary", "")
            description = news.get("content", {}).get("description", "")
            url = news.get("content", {}).get("canonicalUrl", {}).get("url", "")
            news_list.append(
                f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
            )
    if not news_list:
        print(f"No news found for company that searched with {ticker} ticker.")
        return f"No news found for company that searched with {ticker} ticker."
    return "\n\n".join(news_list)


@yfinance_server.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
)
async def get_stock_actions(ticker: str) -> str:
    """Get stock dividends and stock splits for a given ticker symbol"""
    try:
        company = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error: getting stock actions for {ticker}: {e}")
        return f"Error: getting stock actions for {ticker}: {e}"
    actions_df = company.actions
    actions_df = actions_df.reset_index(names="Date")
    return actions_df.to_json(orient="records", date_format="iso")


@yfinance_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    """Get financial statement for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting financial statement for {ticker}: {e}")
        return f"Error: getting financial statement for {ticker}: {e}"

    if financial_type == FinancialType.income_stmt:
        financial_statement = company.income_stmt
    elif financial_type == FinancialType.quarterly_income_stmt:
        financial_statement = company.quarterly_income_stmt
    elif financial_type == FinancialType.balance_sheet:
        financial_statement = company.balance_sheet
    elif financial_type == FinancialType.quarterly_balance_sheet:
        financial_statement = company.quarterly_balance_sheet
    elif financial_type == FinancialType.cashflow:
        financial_statement = company.cashflow
    elif financial_type == FinancialType.quarterly_cashflow:
        financial_statement = company.quarterly_cashflow
    else:
        return f"Error: invalid financial type {financial_type}. Please use one of the following: {FinancialType.income_stmt}, {FinancialType.quarterly_income_stmt}, {FinancialType.balance_sheet}, {FinancialType.quarterly_balance_sheet}, {FinancialType.cashflow}, {FinancialType.quarterly_cashflow}."

    # Create a list to store all the json objects
    result = []

    # Loop through each column (date)
    for column in financial_statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
        else:
            date_str = str(column)

        # Create a dictionary for each date
        date_obj = {"date": date_str}

        # Add each metric as a key-value pair
        for index, value in financial_statement[column].items():
            # Add the value, handling NaN values
            date_obj[index] = None if pd.isna(value) else value

        result.append(date_obj)

    return json.dumps(result)


@yfinance_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting holder info for {ticker}: {e}")
        return f"Error: getting holder info for {ticker}: {e}"

    if holder_type == HolderType.major_holders:
        return company.major_holders.reset_index(names="metric").to_json(orient="records")
    elif holder_type == HolderType.institutional_holders:
        return company.institutional_holders.to_json(orient="records")
    elif holder_type == HolderType.mutualfund_holders:
        return company.mutualfund_holders.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_transactions:
        return company.insider_transactions.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_purchases:
        return company.insider_purchases.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_roster_holders:
        return company.insider_roster_holders.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders}, {HolderType.institutional_holders}, {HolderType.mutualfund_holders}, {HolderType.insider_transactions}, {HolderType.insider_purchases}, {HolderType.insider_roster_holders}."


@yfinance_server.tool(
    name="get_option_expiration_dates",
    description="""Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    """Fetch the available options expiration dates for a given ticker symbol."""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option expiration dates for {ticker}: {e}")
        return f"Error: getting option expiration dates for {ticker}: {e}"
    return json.dumps(company.options)


@yfinance_server.tool(
    name="get_option_chain",
    description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.

Args:
    ticker: str
        The ticker symbol of the stock to get option chain for, e.g. "AAPL"
    expiration_date: str
        The expiration date for the options chain (format: 'YYYY-MM-DD')
    option_type: str
        The type of option to fetch ('calls' or 'puts')
""",
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    """Fetch the option chain for a given ticker symbol, expiration date, and option type.

    Args:
        ticker: The ticker symbol of the stock
        expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: The type of option to fetch ('calls' or 'puts')

    Returns:
        str: JSON string containing the option chain data
    """

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option chain for {ticker}: {e}")
        return f"Error: getting option chain for {ticker}: {e}"

    # Check if the expiration date is valid
    if expiration_date not in company.options:
        return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."

    # Check if the option type is valid
    if option_type not in ["calls", "puts"]:
        return "Error: Invalid option type. Please use 'calls' or 'puts'."

    # Get the option chain
    option_chain = company.option_chain(expiration_date)
    if option_type == "calls":
        return option_chain.calls.to_json(orient="records", date_format="iso")
    elif option_type == "puts":
        return option_chain.puts.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid option type {option_type}. Please use one of the following: calls, puts."


@yfinance_server.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"
    try:
        if recommendation_type == RecommendationType.recommendations:
            return company.recommendations.to_json(orient="records")
        elif recommendation_type == RecommendationType.upgrades_downgrades:
            # Get the upgrades/downgrades based on the cutoff date
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            # Get the first occurrence (most recent) for each firm
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"


@yfinance_server.tool(
    name="filter_stocks_by_price",
    description="""Filter a list of stocks by price range.

Args:
    tickers: list[str]
        A list of ticker symbols to filter, e.g. ["AAPL", "GOOGL", "MSFT"]
    min_price: float
        The minimum stock price (inclusive). Default is 0.
    max_price: float
        The maximum stock price (inclusive). Default is None (no upper limit).
""",
)
async def filter_stocks_by_price(
    tickers: list[str], min_price: float = 0, max_price: float = None
) -> str:
    """Filter a list of stocks by price range"""
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if current_price is None:
                continue
                
            # Check price range
            if current_price >= min_price:
                if max_price is None or current_price <= max_price:
                    results.append({
                        "ticker": ticker,
                        "name": info.get("shortName", "N/A"),
                        "currentPrice": current_price,
                        "currency": info.get("currency", "USD")
                    })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


@yfinance_server.tool(
    name="filter_stocks_by_market_cap",
    description="""Filter a list of stocks by market capitalization category.

Args:
    tickers: list[str]
        A list of ticker symbols to filter, e.g. ["AAPL", "GOOGL", "MSFT"]
    market_cap_category: str
        The market cap category to filter by. Options:
        - "mega": > $200 billion
        - "large": $10 billion - $200 billion
        - "mid": $2 billion - $10 billion
        - "small": $300 million - $2 billion
        - "micro": < $300 million
""",
)
async def filter_stocks_by_market_cap(
    tickers: list[str], market_cap_category: str
) -> str:
    """Filter a list of stocks by market capitalization category"""
    
    # Define market cap ranges in billions
    cap_ranges = {
        "mega": (200e9, float("inf")),
        "large": (10e9, 200e9),
        "mid": (2e9, 10e9),
        "small": (300e6, 2e9),
        "micro": (0, 300e6)
    }
    
    if market_cap_category not in cap_ranges:
        return f"Error: Invalid market cap category '{market_cap_category}'. Use one of: mega, large, mid, small, micro"
    
    min_cap, max_cap = cap_ranges[market_cap_category]
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            market_cap = info.get("marketCap")
            
            if market_cap is None:
                continue
                
            if min_cap <= market_cap < max_cap:
                results.append({
                    "ticker": ticker,
                    "name": info.get("shortName", "N/A"),
                    "marketCap": market_cap,
                    "marketCapFormatted": f"${market_cap / 1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap / 1e6:.2f}M",
                    "category": market_cap_category
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


@yfinance_server.tool(
    name="filter_stocks_by_pe_ratio",
    description="""Filter a list of stocks by P/E (Price-to-Earnings) ratio range.

Args:
    tickers: list[str]
        A list of ticker symbols to filter, e.g. ["AAPL", "GOOGL", "MSFT"]
    min_pe: float
        The minimum P/E ratio (inclusive). Default is 0.
    max_pe: float
        The maximum P/E ratio (inclusive). Default is None (no upper limit).
    use_forward_pe: bool
        If True, use forward P/E ratio. If False, use trailing P/E ratio. Default is False.
""",
)
async def filter_stocks_by_pe_ratio(
    tickers: list[str], min_pe: float = 0, max_pe: float = None, use_forward_pe: bool = False
) -> str:
    """Filter a list of stocks by P/E ratio range"""
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            if use_forward_pe:
                pe_ratio = info.get("forwardPE")
                pe_type = "forwardPE"
            else:
                pe_ratio = info.get("trailingPE")
                pe_type = "trailingPE"
            
            if pe_ratio is None:
                continue
                
            # Check P/E range
            if pe_ratio >= min_pe:
                if max_pe is None or pe_ratio <= max_pe:
                    results.append({
                        "ticker": ticker,
                        "name": info.get("shortName", "N/A"),
                        "peRatio": round(pe_ratio, 2),
                        "peType": pe_type,
                        "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice")
                    })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


@yfinance_server.tool(
    name="filter_stocks_by_dividend_yield",
    description="""Filter a list of stocks by dividend yield category.

Args:
    tickers: list[str]
        A list of ticker symbols to filter, e.g. ["AAPL", "GOOGL", "MSFT"]
    dividend_category: str
        The dividend yield category to filter by. Options:
        - "high_yield": > 4% dividend yield
        - "medium_yield": 2% - 4% dividend yield
        - "low_yield": 0% - 2% dividend yield (but has dividend)
        - "no_dividend": Stocks with no dividend
""",
)
async def filter_stocks_by_dividend_yield(
    tickers: list[str], dividend_category: str
) -> str:
    """Filter a list of stocks by dividend yield category"""
    
    # Define dividend yield ranges
    yield_ranges = {
        "high_yield": (0.04, float("inf")),
        "medium_yield": (0.02, 0.04),
        "low_yield": (0.0001, 0.02),
        "no_dividend": (0, 0.0001)
    }
    
    if dividend_category not in yield_ranges:
        return f"Error: Invalid dividend category '{dividend_category}'. Use one of: high_yield, medium_yield, low_yield, no_dividend"
    
    min_yield, max_yield = yield_ranges[dividend_category]
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            dividend_yield = info.get("dividendYield") or 0
            
            if min_yield <= dividend_yield < max_yield:
                results.append({
                    "ticker": ticker,
                    "name": info.get("shortName", "N/A"),
                    "dividendYield": f"{dividend_yield * 100:.2f}%",
                    "dividendRate": info.get("dividendRate", 0),
                    "category": dividend_category
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


@yfinance_server.tool(
    name="screen_stocks",
    description="""Screen multiple stocks based on various criteria. All criteria are optional - only specify the ones you want to filter by.

Args:
    tickers: list[str]
        A list of ticker symbols to screen, e.g. ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    min_price: float
        Minimum stock price. Default is None (no filter).
    max_price: float
        Maximum stock price. Default is None (no filter).
    min_market_cap: float
        Minimum market cap in billions (e.g., 10 for $10B). Default is None (no filter).
    max_market_cap: float
        Maximum market cap in billions. Default is None (no filter).
    min_pe: float
        Minimum P/E ratio. Default is None (no filter).
    max_pe: float
        Maximum P/E ratio. Default is None (no filter).
    min_dividend_yield: float
        Minimum dividend yield as percentage (e.g., 2 for 2%). Default is None (no filter).
    sector: str
        Filter by sector (e.g., "Technology", "Healthcare"). Default is None (no filter).
""",
)
async def screen_stocks(
    tickers: list[str],
    min_price: float = None,
    max_price: float = None,
    min_market_cap: float = None,
    max_market_cap: float = None,
    min_pe: float = None,
    max_pe: float = None,
    min_dividend_yield: float = None,
    sector: str = None
) -> str:
    """Screen multiple stocks based on various criteria"""
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            # Get stock data
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            dividend_yield = info.get("dividendYield") or 0
            stock_sector = info.get("sector", "")
            
            # Apply filters
            if min_price is not None and (current_price is None or current_price < min_price):
                continue
            if max_price is not None and (current_price is None or current_price > max_price):
                continue
            if min_market_cap is not None and (market_cap is None or market_cap < min_market_cap * 1e9):
                continue
            if max_market_cap is not None and (market_cap is None or market_cap > max_market_cap * 1e9):
                continue
            if min_pe is not None and (pe_ratio is None or pe_ratio < min_pe):
                continue
            if max_pe is not None and (pe_ratio is None or pe_ratio > max_pe):
                continue
            if min_dividend_yield is not None and dividend_yield * 100 < min_dividend_yield:
                continue
            if sector is not None and sector.lower() not in stock_sector.lower():
                continue
            
            # Stock passed all filters
            results.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "currentPrice": current_price,
                "marketCap": f"${market_cap / 1e9:.2f}B" if market_cap and market_cap >= 1e9 else f"${market_cap / 1e6:.2f}M" if market_cap else "N/A",
                "peRatio": round(pe_ratio, 2) if pe_ratio else "N/A",
                "dividendYield": f"{dividend_yield * 100:.2f}%" if dividend_yield else "0%",
                "sector": stock_sector or "N/A"
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


@yfinance_server.tool(
    name="get_stocks_summary",
    description="""Get a summary comparison of multiple stocks with key metrics for quick analysis.

Args:
    tickers: list[str]
        A list of ticker symbols to compare, e.g. ["AAPL", "GOOGL", "MSFT"]
""",
)
async def get_stocks_summary(tickers: list[str]) -> str:
    """Get a summary comparison of multiple stocks with key metrics"""
    results = []
    
    for ticker in tickers:
        try:
            company = yf.Ticker(ticker)
            info = company.info
            
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            market_cap = info.get("marketCap")
            
            results.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "currentPrice": current_price,
                "currency": info.get("currency", "USD"),
                "marketCap": f"${market_cap / 1e9:.2f}B" if market_cap and market_cap >= 1e9 else f"${market_cap / 1e6:.2f}M" if market_cap else "N/A",
                "peRatio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
                "forwardPE": round(info.get("forwardPE"), 2) if info.get("forwardPE") else "N/A",
                "pegRatio": round(info.get("pegRatio"), 2) if info.get("pegRatio") else "N/A",
                "dividendYield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "0%",
                "52WeekHigh": info.get("fiftyTwoWeekHigh"),
                "52WeekLow": info.get("fiftyTwoWeekLow"),
                "50DayAverage": round(info.get("fiftyDayAverage"), 2) if info.get("fiftyDayAverage") else "N/A",
                "200DayAverage": round(info.get("twoHundredDayAverage"), 2) if info.get("twoHundredDayAverage") else "N/A",
                "beta": round(info.get("beta"), 2) if info.get("beta") else "N/A",
                "profitMargins": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get("profitMargins") else "N/A",
                "revenueGrowth": f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get("revenueGrowth") else "N/A"
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return json.dumps(results)


# ==================== TECHNICAL ANALYSIS TOOLS ====================

@yfinance_server.tool(
    name="get_moving_averages",
    description="""Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    periods: list[int]
        List of periods to calculate, e.g. [20, 50, 200]. Default is [20, 50, 200].
    ma_type: str
        Type of moving average: "sma", "ema", or "both". Default is "both".
""",
)
async def get_moving_averages(
    ticker: str, periods: list[int] = [20, 50, 200], ma_type: str = "both"
) -> str:
    """Calculate moving averages for a stock"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period="1y")
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        close_prices = hist['Close']
        current_price = close_prices.iloc[-1]
        
        result = {
            "ticker": ticker,
            "currentPrice": round(current_price, 2),
            "date": hist.index[-1].strftime("%Y-%m-%d"),
            "movingAverages": {}
        }
        
        for period in periods:
            ma_data = {}
            if ma_type in ["sma", "both"]:
                sma = close_prices.rolling(window=period).mean().iloc[-1]
                ma_data["SMA"] = round(sma, 2)
                ma_data["priceVsSMA"] = f"{((current_price - sma) / sma * 100):.2f}%"
            
            if ma_type in ["ema", "both"]:
                ema = close_prices.ewm(span=period, adjust=False).mean().iloc[-1]
                ma_data["EMA"] = round(ema, 2)
                ma_data["priceVsEMA"] = f"{((current_price - ema) / ema * 100):.2f}%"
            
            result["movingAverages"][f"{period}day"] = ma_data
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating moving averages for {ticker}: {e}"


@yfinance_server.tool(
    name="get_rsi",
    description="""Calculate the Relative Strength Index (RSI) for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    period: int
        RSI calculation period. Default is 14.
""",
)
async def get_rsi(ticker: str, period: int = 14) -> str:
    """Calculate RSI for a stock"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period="3mo")
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        rsi_values = calculate_rsi(hist['Close'], period)
        current_rsi = rsi_values.iloc[-1]
        
        # Determine RSI signal
        if current_rsi >= 70:
            signal = "Overbought"
        elif current_rsi <= 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        
        result = {
            "ticker": ticker,
            "period": period,
            "currentRSI": round(current_rsi, 2),
            "signal": signal,
            "date": hist.index[-1].strftime("%Y-%m-%d"),
            "recentRSI": [round(v, 2) for v in rsi_values.tail(5).tolist()]
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating RSI for {ticker}: {e}"


@yfinance_server.tool(
    name="get_macd",
    description="""Calculate the MACD (Moving Average Convergence Divergence) for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    fast_period: int
        Fast EMA period. Default is 12.
    slow_period: int
        Slow EMA period. Default is 26.
    signal_period: int
        Signal line period. Default is 9.
""",
)
async def get_macd(
    ticker: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> str:
    """Calculate MACD for a stock"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period="6mo")
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        macd_line, signal_line, histogram = calculate_macd(
            hist['Close'], fast_period, slow_period, signal_period
        )
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # Determine signal
        if current_macd > current_signal and prev_hist < 0 and current_hist > 0:
            signal = "Bullish Crossover"
        elif current_macd < current_signal and prev_hist > 0 and current_hist < 0:
            signal = "Bearish Crossover"
        elif current_macd > current_signal:
            signal = "Bullish"
        else:
            signal = "Bearish"
        
        result = {
            "ticker": ticker,
            "parameters": f"MACD({fast_period},{slow_period},{signal_period})",
            "macdLine": round(current_macd, 4),
            "signalLine": round(current_signal, 4),
            "histogram": round(current_hist, 4),
            "signal": signal,
            "date": hist.index[-1].strftime("%Y-%m-%d")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating MACD for {ticker}: {e}"


@yfinance_server.tool(
    name="get_bollinger_bands",
    description="""Calculate Bollinger Bands for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    period: int
        SMA period for middle band. Default is 20.
    std_dev: int
        Number of standard deviations for bands. Default is 2.
""",
)
async def get_bollinger_bands(ticker: str, period: int = 20, std_dev: int = 2) -> str:
    """Calculate Bollinger Bands for a stock"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period="3mo")
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        upper, middle, lower = calculate_bollinger_bands(hist['Close'], period, std_dev)
        current_price = hist['Close'].iloc[-1]
        
        # Calculate %B (position within bands)
        percent_b = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        
        # Determine signal
        if current_price > upper.iloc[-1]:
            signal = "Above Upper Band (Overbought)"
        elif current_price < lower.iloc[-1]:
            signal = "Below Lower Band (Oversold)"
        else:
            signal = "Within Bands"
        
        result = {
            "ticker": ticker,
            "parameters": f"BB({period},{std_dev})",
            "currentPrice": round(current_price, 2),
            "upperBand": round(upper.iloc[-1], 2),
            "middleBand": round(middle.iloc[-1], 2),
            "lowerBand": round(lower.iloc[-1], 2),
            "percentB": round(percent_b * 100, 2),
            "bandwidth": round((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100, 2),
            "signal": signal,
            "date": hist.index[-1].strftime("%Y-%m-%d")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating Bollinger Bands for {ticker}: {e}"


@yfinance_server.tool(
    name="get_support_resistance_levels",
    description="""Calculate support and resistance levels for a stock based on recent price action.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    period: str
        Historical period to analyze. Default is "6mo".
""",
)
async def get_support_resistance_levels(ticker: str, period: str = "6mo") -> str:
    """Calculate support and resistance levels"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period=period)
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        high = hist['High'].max()
        low = hist['Low'].min()
        
        # Pivot point calculation
        typical_price = (hist['High'].iloc[-1] + hist['Low'].iloc[-1] + hist['Close'].iloc[-1]) / 3
        
        # Support and Resistance levels
        r1 = 2 * typical_price - hist['Low'].iloc[-1]
        r2 = typical_price + (hist['High'].iloc[-1] - hist['Low'].iloc[-1])
        s1 = 2 * typical_price - hist['High'].iloc[-1]
        s2 = typical_price - (hist['High'].iloc[-1] - hist['Low'].iloc[-1])
        
        # Key levels (52-week high/low, recent highs/lows)
        week_52_high = hist['High'].max()
        week_52_low = hist['Low'].min()
        
        result = {
            "ticker": ticker,
            "currentPrice": round(current_price, 2),
            "pivotPoint": round(typical_price, 2),
            "resistance": {
                "R1": round(r1, 2),
                "R2": round(r2, 2),
                "52WeekHigh": round(week_52_high, 2)
            },
            "support": {
                "S1": round(s1, 2),
                "S2": round(s2, 2),
                "52WeekLow": round(week_52_low, 2)
            },
            "priceRange": {
                "high": round(high, 2),
                "low": round(low, 2),
                "range": round(high - low, 2),
                "currentPositionPercent": round((current_price - low) / (high - low) * 100, 2)
            },
            "date": hist.index[-1].strftime("%Y-%m-%d")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating support/resistance for {ticker}: {e}"


@yfinance_server.tool(
    name="get_technical_summary",
    description="""Get a comprehensive technical analysis summary for a stock including multiple indicators.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_technical_summary(ticker: str) -> str:
    """Get comprehensive technical analysis summary"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period="1y")
        
        if hist.empty:
            return f"Error: No historical data found for {ticker}"
        
        close = hist['Close']
        current_price = close.iloc[-1]
        
        # Calculate indicators
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
        
        rsi = calculate_rsi(close, 14).iloc[-1]
        macd_line, signal_line, histogram = calculate_macd(close)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(close)
        
        # Volume analysis
        avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Trend analysis
        signals = []
        bullish_count = 0
        bearish_count = 0
        
        if current_price > sma_20:
            bullish_count += 1
            signals.append("Above SMA20 (Bullish)")
        else:
            bearish_count += 1
            signals.append("Below SMA20 (Bearish)")
            
        if current_price > sma_50:
            bullish_count += 1
            signals.append("Above SMA50 (Bullish)")
        else:
            bearish_count += 1
            signals.append("Below SMA50 (Bearish)")
            
        if current_price > sma_200:
            bullish_count += 1
            signals.append("Above SMA200 (Bullish)")
        else:
            bearish_count += 1
            signals.append("Below SMA200 (Bearish)")
            
        if rsi < 30:
            signals.append("RSI Oversold (Potential Bullish)")
        elif rsi > 70:
            signals.append("RSI Overbought (Potential Bearish)")
            
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            bullish_count += 1
            signals.append("MACD Bullish")
        else:
            bearish_count += 1
            signals.append("MACD Bearish")
        
        overall_signal = "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral"
        
        result = {
            "ticker": ticker,
            "currentPrice": round(current_price, 2),
            "date": hist.index[-1].strftime("%Y-%m-%d"),
            "movingAverages": {
                "SMA20": round(sma_20, 2),
                "SMA50": round(sma_50, 2),
                "SMA200": round(sma_200, 2),
                "EMA12": round(ema_12, 2),
                "EMA26": round(ema_26, 2)
            },
            "oscillators": {
                "RSI14": round(rsi, 2),
                "MACD": round(macd_line.iloc[-1], 4),
                "MACDSignal": round(signal_line.iloc[-1], 4),
                "MACDHistogram": round(histogram.iloc[-1], 4)
            },
            "bollingerBands": {
                "upper": round(upper_bb.iloc[-1], 2),
                "middle": round(middle_bb.iloc[-1], 2),
                "lower": round(lower_bb.iloc[-1], 2)
            },
            "volume": {
                "current": int(current_volume),
                "average20Day": int(avg_volume),
                "volumeRatio": round(volume_ratio, 2)
            },
            "signals": signals,
            "overallSignal": overall_signal,
            "bullishIndicators": bullish_count,
            "bearishIndicators": bearish_count
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error generating technical summary for {ticker}: {e}"


# ==================== PORTFOLIO MANAGEMENT TOOLS ====================

@yfinance_server.tool(
    name="calculate_portfolio_value",
    description="""Calculate the current value of a portfolio.

Args:
    holdings: dict
        Dictionary of ticker symbols and quantities, e.g. {"AAPL": 10, "GOOGL": 5, "MSFT": 15}
""",
)
async def calculate_portfolio_value(holdings: dict) -> str:
    """Calculate the current value of a portfolio"""
    try:
        portfolio_value = 0
        positions = []
        
        for ticker, quantity in holdings.items():
            company = get_cached_ticker(ticker)
            info = company.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            position_value = current_price * quantity
            portfolio_value += position_value
            
            positions.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "quantity": quantity,
                "currentPrice": round(current_price, 2),
                "positionValue": round(position_value, 2),
                "currency": info.get("currency", "USD")
            })
        
        # Calculate weights
        for position in positions:
            position["weight"] = f"{(position['positionValue'] / portfolio_value * 100):.2f}%"
        
        result = {
            "totalValue": round(portfolio_value, 2),
            "numberOfPositions": len(positions),
            "positions": positions,
            "calculatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating portfolio value: {e}"


@yfinance_server.tool(
    name="get_portfolio_allocation",
    description="""Get sector and industry breakdown of a portfolio.

Args:
    holdings: dict
        Dictionary of ticker symbols and quantities, e.g. {"AAPL": 10, "GOOGL": 5}
""",
)
async def get_portfolio_allocation(holdings: dict) -> str:
    """Get sector and industry breakdown of a portfolio"""
    try:
        sector_allocation = {}
        industry_allocation = {}
        total_value = 0
        
        positions = []
        
        for ticker, quantity in holdings.items():
            company = get_cached_ticker(ticker)
            info = company.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            position_value = current_price * quantity
            total_value += position_value
            
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")
            
            sector_allocation[sector] = sector_allocation.get(sector, 0) + position_value
            industry_allocation[industry] = industry_allocation.get(industry, 0) + position_value
            
            positions.append({
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "value": position_value
            })
        
        # Convert to percentages
        sector_breakdown = {
            sector: {"value": round(value, 2), "weight": f"{(value / total_value * 100):.2f}%"}
            for sector, value in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True)
        }
        
        industry_breakdown = {
            industry: {"value": round(value, 2), "weight": f"{(value / total_value * 100):.2f}%"}
            for industry, value in sorted(industry_allocation.items(), key=lambda x: x[1], reverse=True)
        }
        
        result = {
            "totalValue": round(total_value, 2),
            "sectorAllocation": sector_breakdown,
            "industryAllocation": industry_breakdown,
            "diversificationScore": len(sector_allocation),
            "topSector": max(sector_allocation, key=sector_allocation.get) if sector_allocation else "N/A"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating portfolio allocation: {e}"


@yfinance_server.tool(
    name="calculate_portfolio_returns",
    description="""Calculate portfolio returns over a specified period.

Args:
    holdings: dict
        Dictionary of ticker symbols and quantities, e.g. {"AAPL": 10, "GOOGL": 5}
    period: str
        Period to calculate returns for: "1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd". Default is "1mo".
""",
)
async def calculate_portfolio_returns(holdings: dict, period: str = "1mo") -> str:
    """Calculate portfolio returns over a period"""
    try:
        total_start_value = 0
        total_current_value = 0
        position_returns = []
        
        for ticker, quantity in holdings.items():
            company = get_cached_ticker(ticker)
            hist = company.history(period=period)
            
            if hist.empty:
                continue
                
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            
            start_value = start_price * quantity
            end_value = end_price * quantity
            position_return = (end_price - start_price) / start_price * 100
            
            total_start_value += start_value
            total_current_value += end_value
            
            position_returns.append({
                "ticker": ticker,
                "quantity": quantity,
                "startPrice": round(start_price, 2),
                "endPrice": round(end_price, 2),
                "startValue": round(start_value, 2),
                "endValue": round(end_value, 2),
                "return": f"{position_return:.2f}%",
                "gainLoss": round(end_value - start_value, 2)
            })
        
        portfolio_return = (total_current_value - total_start_value) / total_start_value * 100 if total_start_value > 0 else 0
        
        result = {
            "period": period,
            "startValue": round(total_start_value, 2),
            "currentValue": round(total_current_value, 2),
            "totalReturn": f"{portfolio_return:.2f}%",
            "totalGainLoss": round(total_current_value - total_start_value, 2),
            "positions": sorted(position_returns, key=lambda x: float(x['return'].rstrip('%')), reverse=True)
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating portfolio returns: {e}"


@yfinance_server.tool(
    name="get_portfolio_risk_metrics",
    description="""Calculate risk metrics for a portfolio including volatility, beta, and Sharpe ratio.

Args:
    holdings: dict
        Dictionary of ticker symbols and quantities, e.g. {"AAPL": 10, "GOOGL": 5}
    benchmark: str
        Benchmark ticker for beta calculation. Default is "SPY".
""",
)
async def get_portfolio_risk_metrics(holdings: dict, benchmark: str = "SPY") -> str:
    """Calculate portfolio risk metrics"""
    try:
        # Get historical data for all holdings
        weights = []
        total_value = 0
        
        # First pass: calculate total value
        for ticker, quantity in holdings.items():
            company = get_cached_ticker(ticker)
            info = company.info
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            total_value += price * quantity
        
        # Second pass: calculate weighted returns
        stock_data = {}
        for ticker, quantity in holdings.items():
            company = get_cached_ticker(ticker)
            hist = company.history(period="1y")
            if hist.empty:
                continue
            
            info = company.info
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            weight = (price * quantity) / total_value
            weights.append(weight)
            
            returns = hist['Close'].pct_change().dropna()
            stock_data[ticker] = {
                "returns": returns,
                "weight": weight,
                "volatility": returns.std() * np.sqrt(252),
                "beta": info.get("beta", 1.0)
            }
        
        # Get benchmark data
        benchmark_company = get_cached_ticker(benchmark)
        benchmark_hist = benchmark_company.history(period="1y")
        _benchmark_returns = benchmark_hist['Close'].pct_change().dropna()  # Reserved for future correlation calculation
        
        # Calculate portfolio metrics
        portfolio_daily_returns = sum(
            stock_data[ticker]["returns"] * stock_data[ticker]["weight"]
            for ticker in stock_data
        )
        
        portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
        portfolio_annual_return = portfolio_daily_returns.mean() * 252
        
        # Calculate portfolio beta
        portfolio_beta = sum(
            stock_data[ticker]["beta"] * stock_data[ticker]["weight"]
            for ticker in stock_data
            if stock_data[ticker]["beta"]
        )
        
        # Sharpe Ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = total_value * (portfolio_daily_returns.quantile(0.05))
        
        result = {
            "totalValue": round(total_value, 2),
            "riskMetrics": {
                "annualizedVolatility": f"{portfolio_volatility * 100:.2f}%",
                "portfolioBeta": round(portfolio_beta, 2),
                "sharpeRatio": round(sharpe_ratio, 2),
                "expectedAnnualReturn": f"{portfolio_annual_return * 100:.2f}%",
                "valueAtRisk95": round(var_95, 2),
                "maxDrawdown": f"{portfolio_daily_returns.min() * 100:.2f}%"
            },
            "benchmark": benchmark,
            "individualStockRisk": {
                ticker: {
                    "weight": f"{data['weight'] * 100:.2f}%",
                    "volatility": f"{data['volatility'] * 100:.2f}%",
                    "beta": round(data['beta'], 2) if data['beta'] else "N/A"
                }
                for ticker, data in stock_data.items()
            }
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating portfolio risk metrics: {e}"


@yfinance_server.tool(
    name="get_correlation_matrix",
    description="""Calculate the correlation matrix for a list of stocks.

Args:
    tickers: list[str]
        List of ticker symbols to analyze, e.g. ["AAPL", "GOOGL", "MSFT", "AMZN"]
    period: str
        Historical period for correlation calculation. Default is "1y".
""",
)
async def get_correlation_matrix(tickers: list[str], period: str = "1y") -> str:
    """Calculate correlation matrix for multiple stocks"""
    try:
        returns_data = {}
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            hist = company.history(period=period)
            if not hist.empty:
                returns_data[ticker] = hist['Close'].pct_change().dropna()
        
        # Create DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Find highly correlated pairs
        high_correlation_pairs = []
        for i, ticker1 in enumerate(corr_matrix.columns):
            for j, ticker2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_value = corr_matrix.loc[ticker1, ticker2]
                    if abs(corr_value) > 0.7:
                        high_correlation_pairs.append({
                            "pair": f"{ticker1}-{ticker2}",
                            "correlation": round(corr_value, 3),
                            "relationship": "Strongly Positive" if corr_value > 0.7 else "Strongly Negative"
                        })
        
        result = {
            "tickers": tickers,
            "period": period,
            "correlationMatrix": {
                ticker: {t: round(corr_matrix.loc[ticker, t], 3) for t in tickers}
                for ticker in tickers
            },
            "highCorrelationPairs": high_correlation_pairs,
            "averageCorrelation": round(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(), 3),
            "diversificationInsight": "Well diversified" if len(high_correlation_pairs) == 0 else f"Consider diversifying - {len(high_correlation_pairs)} highly correlated pairs found"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating correlation matrix: {e}"


# ==================== EARNINGS & EVENTS TOOLS ====================

@yfinance_server.tool(
    name="get_earnings_calendar",
    description="""Get upcoming and recent earnings dates for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_earnings_calendar(ticker: str) -> str:
    """Get earnings calendar for a stock"""
    try:
        company = get_cached_ticker(ticker)
        
        result = {
            "ticker": ticker
        }
        
        # Get calendar info
        try:
            calendar = company.calendar
            if calendar is not None and not calendar.empty:
                result["calendar"] = calendar.to_dict()
        except Exception:
            result["calendar"] = None
        
        # Get earnings dates
        try:
            earnings_dates = company.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                earnings_df = earnings_dates.reset_index()
                earnings_df.columns = earnings_df.columns.astype(str)
                result["earningsDates"] = json.loads(earnings_df.head(10).to_json(orient="records", date_format="iso"))
        except Exception:
            result["earningsDates"] = None
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting earnings calendar for {ticker}: {e}"


@yfinance_server.tool(
    name="get_earnings_history",
    description="""Get historical earnings (EPS) data and compare actual vs estimates.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_earnings_history(ticker: str) -> str:
    """Get historical earnings data"""
    try:
        company = get_cached_ticker(ticker)
        
        result = {
            "ticker": ticker
        }
        
        # Get earnings history
        try:
            earnings_hist = company.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                result["earningsHistory"] = json.loads(earnings_hist.reset_index().to_json(orient="records", date_format="iso"))
        except Exception:
            result["earningsHistory"] = None
        
        # Get quarterly earnings
        try:
            quarterly_earnings = company.quarterly_earnings
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                result["quarterlyEarnings"] = json.loads(quarterly_earnings.reset_index().to_json(orient="records"))
        except Exception:
            result["quarterlyEarnings"] = None
        
        # Get annual earnings
        try:
            earnings = company.earnings
            if earnings is not None and not earnings.empty:
                result["annualEarnings"] = json.loads(earnings.reset_index().to_json(orient="records"))
        except Exception:
            result["annualEarnings"] = None
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting earnings history for {ticker}: {e}"


@yfinance_server.tool(
    name="get_analyst_estimates",
    description="""Get analyst earnings and revenue estimates for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_analyst_estimates(ticker: str) -> str:
    """Get analyst estimates for a stock"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A")
        }
        
        # Earnings estimates
        try:
            earnings_estimate = company.earnings_estimate
            if earnings_estimate is not None and not earnings_estimate.empty:
                result["earningsEstimate"] = json.loads(earnings_estimate.to_json())
        except Exception:
            result["earningsEstimate"] = None
        
        # Revenue estimates
        try:
            revenue_estimate = company.revenue_estimate
            if revenue_estimate is not None and not revenue_estimate.empty:
                result["revenueEstimate"] = json.loads(revenue_estimate.to_json())
        except Exception:
            result["revenueEstimate"] = None
        
        # EPS trend
        try:
            eps_trend = company.eps_trend
            if eps_trend is not None and not eps_trend.empty:
                result["epsTrend"] = json.loads(eps_trend.to_json())
        except Exception:
            result["epsTrend"] = None
        
        # Growth estimates
        try:
            growth_estimates = company.growth_estimates
            if growth_estimates is not None and not growth_estimates.empty:
                result["growthEstimates"] = json.loads(growth_estimates.to_json())
        except Exception:
            result["growthEstimates"] = None
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting analyst estimates for {ticker}: {e}"


@yfinance_server.tool(
    name="get_upcoming_events",
    description="""Get upcoming events for a list of stocks including earnings dates, ex-dividend dates, etc.

Args:
    tickers: list[str]
        List of ticker symbols, e.g. ["AAPL", "GOOGL", "MSFT"]
""",
)
async def get_upcoming_events(tickers: list[str]) -> str:
    """Get upcoming events for multiple stocks"""
    try:
        events = []
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            info = company.info
            
            stock_events = {
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "events": []
            }
            
            # Earnings date
            try:
                calendar = company.calendar
                if calendar is not None:
                    if hasattr(calendar, 'get'):
                        earnings_date = calendar.get('Earnings Date')
                        if earnings_date:
                            stock_events["events"].append({
                                "type": "Earnings",
                                "date": str(earnings_date)
                            })
            except Exception:
                pass
            
            # Ex-dividend date
            ex_div_date = info.get("exDividendDate")
            if ex_div_date:
                stock_events["events"].append({
                    "type": "Ex-Dividend",
                    "date": datetime.fromtimestamp(ex_div_date).strftime("%Y-%m-%d"),
                    "dividendRate": info.get("dividendRate", "N/A")
                })
            
            events.append(stock_events)
        
        return json.dumps(events)
    except Exception as e:
        return f"Error getting upcoming events: {e}"


# ==================== PEER COMPARISON & INDUSTRY ANALYSIS ====================

@yfinance_server.tool(
    name="get_sector_performance",
    description="""Get performance metrics for major market sectors using sector ETFs.
""",
)
async def get_sector_performance() -> str:
    """Get performance of major market sectors"""
    try:
        # Major sector ETFs
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financial": "XLF",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Industrials": "XLI",
            "Communication Services": "XLC"
        }
        
        sector_data = []
        
        for sector_name, etf_ticker in sector_etfs.items():
            company = get_cached_ticker(etf_ticker)
            hist = company.history(period="1y")
            
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            month_ago = hist['Close'].iloc[-22] if len(hist) > 22 else hist['Close'].iloc[0]
            
            sector_data.append({
                "sector": sector_name,
                "etf": etf_ticker,
                "currentPrice": round(current_price, 2),
                "yearToDateReturn": f"{((current_price - start_price) / start_price * 100):.2f}%",
                "monthReturn": f"{((current_price - month_ago) / month_ago * 100):.2f}%",
                "52WeekHigh": round(hist['High'].max(), 2),
                "52WeekLow": round(hist['Low'].min(), 2)
            })
        
        # Sort by YTD return
        sector_data.sort(key=lambda x: float(x['yearToDateReturn'].rstrip('%')), reverse=True)
        
        result = {
            "sectors": sector_data,
            "topPerformer": sector_data[0]["sector"] if sector_data else "N/A",
            "worstPerformer": sector_data[-1]["sector"] if sector_data else "N/A",
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting sector performance: {e}"


@yfinance_server.tool(
    name="get_industry_peers",
    description="""Find peer companies in the same industry as the given stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    limit: int
        Maximum number of peers to return. Default is 10.
""",
)
async def get_industry_peers(ticker: str, limit: int = 10) -> str:
    """Find peer companies in the same industry"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        market_cap = info.get("marketCap", 0)
        
        # Common industry peer mappings
        industry_peers = {
            "Consumer Electronics": ["AAPL", "SONY", "LOGI", "HPQ", "DELL"],
            "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS", "TWTR"],
            "Software - Infrastructure": ["MSFT", "ORCL", "CRM", "NOW", "ADBE"],
            "Software - Application": ["ADBE", "CRM", "INTU", "WDAY", "SPLK"],
            "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM"],
            "Auto Manufacturers": ["TSLA", "F", "GM", "TM", "HMC", "RIVN"],
            "Banks - Diversified": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
            "Drug Manufacturers - General": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
            "Internet Retail": ["AMZN", "EBAY", "ETSY", "W", "CHWY"]
        }
        
        # Get peers from mapping or return message
        peers_list = industry_peers.get(industry, [])
        
        # Remove the input ticker from peers
        peers_list = [p for p in peers_list if p.upper() != ticker.upper()][:limit]
        
        peers_data = []
        for peer_ticker in peers_list:
            try:
                peer_company = get_cached_ticker(peer_ticker)
                peer_info = peer_company.info
                peers_data.append({
                    "ticker": peer_ticker,
                    "name": peer_info.get("shortName", "N/A"),
                    "marketCap": f"${peer_info.get('marketCap', 0) / 1e9:.2f}B",
                    "peRatio": round(peer_info.get("trailingPE", 0), 2) if peer_info.get("trailingPE") else "N/A",
                    "industry": peer_info.get("industry", "N/A")
                })
            except Exception:
                continue
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "sector": sector,
            "industry": industry,
            "marketCap": f"${market_cap / 1e9:.2f}B" if market_cap else "N/A",
            "peers": peers_data,
            "peerCount": len(peers_data)
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error finding industry peers for {ticker}: {e}"


@yfinance_server.tool(
    name="compare_to_peers",
    description="""Compare a stock's key metrics to its peers.

Args:
    ticker: str
        The main ticker to compare
    peer_tickers: list[str]
        List of peer ticker symbols to compare against
    metrics: list[str]
        Metrics to compare. Options: "pe", "pb", "ps", "margin", "growth", "dividend". Default includes all.
""",
)
async def compare_to_peers(
    ticker: str, 
    peer_tickers: list[str], 
    metrics: list[str] = ["pe", "pb", "ps", "margin", "growth", "dividend"]
) -> str:
    """Compare a stock to its peers"""
    try:
        all_tickers = [ticker] + peer_tickers
        comparison_data = []
        
        for t in all_tickers:
            company = get_cached_ticker(t)
            info = company.info
            
            stock_metrics = {
                "ticker": t,
                "name": info.get("shortName", "N/A"),
                "isMainStock": t == ticker
            }
            
            if "pe" in metrics:
                stock_metrics["trailingPE"] = round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A"
                stock_metrics["forwardPE"] = round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A"
            
            if "pb" in metrics:
                stock_metrics["priceToBook"] = round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A"
            
            if "ps" in metrics:
                stock_metrics["priceToSales"] = round(info.get("priceToSalesTrailing12Months", 0), 2) if info.get("priceToSalesTrailing12Months") else "N/A"
            
            if "margin" in metrics:
                stock_metrics["profitMargin"] = f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get("profitMargins") else "N/A"
                stock_metrics["operatingMargin"] = f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get("operatingMargins") else "N/A"
            
            if "growth" in metrics:
                stock_metrics["revenueGrowth"] = f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get("revenueGrowth") else "N/A"
                stock_metrics["earningsGrowth"] = f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get("earningsGrowth") else "N/A"
            
            if "dividend" in metrics:
                stock_metrics["dividendYield"] = f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "0%"
            
            comparison_data.append(stock_metrics)
        
        result = {
            "mainStock": ticker,
            "comparison": comparison_data,
            "peerCount": len(peer_tickers),
            "metricsCompared": metrics
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error comparing to peers: {e}"


# ==================== ADVANCED VALUATION METRICS ====================

@yfinance_server.tool(
    name="get_valuation_ratios",
    description="""Get comprehensive valuation ratios for a stock.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_valuation_ratios(ticker: str) -> str:
    """Get comprehensive valuation ratios"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "priceMultiples": {
                "trailingPE": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
                "forwardPE": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
                "pegRatio": round(info.get("pegRatio", 0), 2) if info.get("pegRatio") else "N/A",
                "priceToBook": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A",
                "priceToSales": round(info.get("priceToSalesTrailing12Months", 0), 2) if info.get("priceToSalesTrailing12Months") else "N/A"
            },
            "enterpriseValue": {
                "enterpriseValue": f"${info.get('enterpriseValue', 0) / 1e9:.2f}B" if info.get("enterpriseValue") else "N/A",
                "evToRevenue": round(info.get("enterpriseToRevenue", 0), 2) if info.get("enterpriseToRevenue") else "N/A",
                "evToEbitda": round(info.get("enterpriseToEbitda", 0), 2) if info.get("enterpriseToEbitda") else "N/A"
            },
            "fundamentals": {
                "marketCap": f"${info.get('marketCap', 0) / 1e9:.2f}B" if info.get("marketCap") else "N/A",
                "bookValue": round(info.get("bookValue", 0), 2) if info.get("bookValue") else "N/A",
                "eps": round(info.get("trailingEps", 0), 2) if info.get("trailingEps") else "N/A",
                "forwardEps": round(info.get("forwardEps", 0), 2) if info.get("forwardEps") else "N/A"
            },
            "profitability": {
                "profitMargin": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get("profitMargins") else "N/A",
                "operatingMargin": f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get("operatingMargins") else "N/A",
                "grossMargin": f"{info.get('grossMargins', 0) * 100:.2f}%" if info.get("grossMargins") else "N/A",
                "returnOnEquity": f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get("returnOnEquity") else "N/A",
                "returnOnAssets": f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get("returnOnAssets") else "N/A"
            }
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting valuation ratios for {ticker}: {e}"


@yfinance_server.tool(
    name="get_dcf_inputs",
    description="""Get inputs needed for DCF (Discounted Cash Flow) valuation.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
""",
)
async def get_dcf_inputs(ticker: str) -> str:
    """Get DCF valuation inputs"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        cashflow = company.cashflow
        
        # Get free cash flow data
        fcf_data = []
        if cashflow is not None and not cashflow.empty:
            for col in cashflow.columns[:4]:  # Last 4 years
                try:
                    operating_cf = cashflow.loc['Operating Cash Flow', col] if 'Operating Cash Flow' in cashflow.index else 0
                    capex = cashflow.loc['Capital Expenditure', col] if 'Capital Expenditure' in cashflow.index else 0
                    fcf = operating_cf + capex  # capex is usually negative
                    fcf_data.append({
                        "year": col.strftime("%Y") if hasattr(col, 'strftime') else str(col),
                        "operatingCashFlow": round(operating_cf / 1e9, 2),
                        "capex": round(capex / 1e9, 2),
                        "freeCashFlow": round(fcf / 1e9, 2)
                    })
                except Exception:
                    continue
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "marketCap": f"${info.get('marketCap', 0) / 1e9:.2f}B",
            "freeCashFlowHistory": fcf_data,
            "growthRates": {
                "revenueGrowth": f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get("revenueGrowth") else "N/A",
                "earningsGrowth": f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get("earningsGrowth") else "N/A"
            },
            "riskMetrics": {
                "beta": round(info.get("beta", 1), 2) if info.get("beta") else 1.0,
                "debtToEquity": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else "N/A"
            },
            "suggestedWACC": "8-12% for mature companies, 12-20% for growth companies",
            "terminalGrowthRate": "2-3% (typically GDP growth rate)"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting DCF inputs for {ticker}: {e}"


@yfinance_server.tool(
    name="calculate_intrinsic_value",
    description="""Calculate estimated intrinsic value using a simplified DCF model.

Args:
    ticker: str
        The ticker symbol, e.g. "AAPL"
    growth_rate: float
        Expected annual growth rate for next 5 years (e.g., 0.10 for 10%). Default is 0.10.
    discount_rate: float
        Discount rate / WACC (e.g., 0.10 for 10%). Default is 0.10.
    terminal_growth: float
        Terminal growth rate (e.g., 0.025 for 2.5%). Default is 0.025.
""",
)
async def calculate_intrinsic_value(
    ticker: str,
    growth_rate: float = 0.10,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025
) -> str:
    """Calculate intrinsic value using DCF"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        cashflow = company.cashflow
        
        # Get most recent free cash flow
        fcf = None
        if cashflow is not None and not cashflow.empty:
            try:
                operating_cf = cashflow.loc['Operating Cash Flow'].iloc[0]
                capex = cashflow.loc['Capital Expenditure'].iloc[0]
                fcf = operating_cf + capex
            except Exception:
                pass
        
        if fcf is None or fcf <= 0:
            return json.dumps({
                "ticker": ticker,
                "error": "Unable to calculate - negative or missing free cash flow"
            })
        
        shares_outstanding = info.get("sharesOutstanding", 1)
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        
        # Project FCF for next 5 years
        projected_fcf = []
        for year in range(1, 6):
            projected = fcf * ((1 + growth_rate) ** year)
            discounted = projected / ((1 + discount_rate) ** year)
            projected_fcf.append({
                "year": year,
                "projectedFCF": round(projected / 1e9, 2),
                "discountedFCF": round(discounted / 1e9, 2)
            })
        
        # Sum of discounted FCF
        total_pv_fcf = sum(p["discountedFCF"] for p in projected_fcf)
        
        # Terminal value
        final_fcf = fcf * ((1 + growth_rate) ** 5)
        terminal_value = final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
        
        # Enterprise value and equity value
        enterprise_value = (total_pv_fcf * 1e9) + pv_terminal
        
        # Get net debt
        total_debt = info.get("totalDebt", 0)
        cash = info.get("totalCash", 0)
        net_debt = total_debt - cash
        
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        # Margin of safety
        upside = (intrinsic_value_per_share - current_price) / current_price * 100
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "currentPrice": round(current_price, 2),
            "assumptions": {
                "baseFCF": f"${fcf / 1e9:.2f}B",
                "growthRate": f"{growth_rate * 100:.1f}%",
                "discountRate": f"{discount_rate * 100:.1f}%",
                "terminalGrowthRate": f"{terminal_growth * 100:.1f}%"
            },
            "projectedFCF": projected_fcf,
            "valuation": {
                "pvOfCashFlows": f"${total_pv_fcf:.2f}B",
                "pvOfTerminalValue": f"${pv_terminal / 1e9:.2f}B",
                "enterpriseValue": f"${enterprise_value / 1e9:.2f}B",
                "netDebt": f"${net_debt / 1e9:.2f}B",
                "equityValue": f"${equity_value / 1e9:.2f}B",
                "intrinsicValuePerShare": round(intrinsic_value_per_share, 2)
            },
            "analysis": {
                "upsideDownside": f"{upside:.1f}%",
                "recommendation": "Undervalued" if upside > 20 else "Fairly Valued" if upside > -20 else "Overvalued"
            }
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating intrinsic value for {ticker}: {e}"


# ==================== CURRENCY & CRYPTO SUPPORT ====================

@yfinance_server.tool(
    name="get_exchange_rate",
    description="""Get currency exchange rate between two currencies.

Args:
    from_currency: str
        Source currency code, e.g. "USD"
    to_currency: str
        Target currency code, e.g. "EUR"
""",
)
async def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get currency exchange rate"""
    try:
        ticker_symbol = f"{from_currency}{to_currency}=X"
        company = yf.Ticker(ticker_symbol)
        hist = company.history(period="5d")
        
        if hist.empty:
            return f"Error: Could not fetch exchange rate for {from_currency}/{to_currency}"
        
        current_rate = hist['Close'].iloc[-1]
        prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
        change = (current_rate - prev_rate) / prev_rate * 100
        
        # Get historical rates
        weekly_hist = company.history(period="1mo")
        week_ago = weekly_hist['Close'].iloc[-5] if len(weekly_hist) > 5 else current_rate
        month_ago = weekly_hist['Close'].iloc[0] if len(weekly_hist) > 0 else current_rate
        
        result = {
            "pair": f"{from_currency}/{to_currency}",
            "rate": round(current_rate, 4),
            "change": f"{change:.2f}%",
            "date": hist.index[-1].strftime("%Y-%m-%d"),
            "historicalRates": {
                "weekAgo": round(week_ago, 4),
                "monthAgo": round(month_ago, 4),
                "weekChange": f"{((current_rate - week_ago) / week_ago * 100):.2f}%",
                "monthChange": f"{((current_rate - month_ago) / month_ago * 100):.2f}%"
            },
            "conversion": {
                f"1 {from_currency}": f"{current_rate:.4f} {to_currency}",
                f"1 {to_currency}": f"{1/current_rate:.4f} {from_currency}"
            }
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting exchange rate: {e}"


@yfinance_server.tool(
    name="get_crypto_price",
    description="""Get cryptocurrency price and information.

Args:
    symbol: str
        Crypto symbol with currency pair, e.g. "BTC-USD", "ETH-USD", "DOGE-USD"
""",
)
async def get_crypto_price(symbol: str) -> str:
    """Get cryptocurrency price"""
    try:
        company = yf.Ticker(symbol)
        hist = company.history(period="1mo")
        info = company.info
        
        if hist.empty:
            return f"Error: Could not fetch data for {symbol}"
        
        current_price = hist['Close'].iloc[-1]
        
        result = {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "currentPrice": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            "dayChange": {
                "open": round(hist['Open'].iloc[-1], 2),
                "high": round(hist['High'].iloc[-1], 2),
                "low": round(hist['Low'].iloc[-1], 2),
                "close": round(hist['Close'].iloc[-1], 2),
                "volume": int(hist['Volume'].iloc[-1])
            },
            "monthStats": {
                "high": round(hist['High'].max(), 2),
                "low": round(hist['Low'].min(), 2),
                "avgVolume": int(hist['Volume'].mean())
            },
            "returns": {
                "day": f"{((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100):.2f}%" if len(hist) > 1 else "N/A",
                "week": f"{((current_price - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7] * 100):.2f}%" if len(hist) > 7 else "N/A",
                "month": f"{((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100):.2f}%"
            },
            "date": hist.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add market cap if available
        if info.get("marketCap"):
            result["marketCap"] = f"${info['marketCap'] / 1e9:.2f}B"
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting crypto price for {symbol}: {e}"


@yfinance_server.tool(
    name="get_crypto_comparison",
    description="""Compare multiple cryptocurrencies.

Args:
    symbols: list[str]
        List of crypto symbols, e.g. ["BTC-USD", "ETH-USD", "SOL-USD"]
""",
)
async def get_crypto_comparison(symbols: list[str]) -> str:
    """Compare multiple cryptocurrencies"""
    try:
        comparison = []
        
        for symbol in symbols:
            company = yf.Ticker(symbol)
            hist = company.history(period="1y")
            info = company.info
            
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            
            comparison.append({
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "currentPrice": round(current_price, 2),
                "marketCap": f"${info.get('marketCap', 0) / 1e9:.2f}B" if info.get("marketCap") else "N/A",
                "yearReturn": f"{((current_price - start_price) / start_price * 100):.2f}%",
                "52WeekHigh": round(hist['High'].max(), 2),
                "52WeekLow": round(hist['Low'].min(), 2),
                "volatility": f"{hist['Close'].pct_change().std() * np.sqrt(365) * 100:.2f}%"
            })
        
        # Sort by market cap or year return
        comparison.sort(key=lambda x: float(x['yearReturn'].rstrip('%')), reverse=True)
        
        return json.dumps({
            "cryptos": comparison,
            "topPerformer": comparison[0]["symbol"] if comparison else "N/A",
            "date": datetime.now().strftime("%Y-%m-%d")
        })
    except Exception as e:
        return f"Error comparing cryptocurrencies: {e}"


# ==================== INDEX & ETF TOOLS ====================

@yfinance_server.tool(
    name="get_major_indices",
    description="""Get current values and performance of major market indices.
""",
)
async def get_major_indices() -> str:
    """Get major market indices"""
    try:
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX",
            "10Y Treasury": "^TNX"
        }
        
        index_data = []
        
        for name, symbol in indices.items():
            company = yf.Ticker(symbol)
            hist = company.history(period="5d")
            
            if hist.empty:
                continue
            
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            
            index_data.append({
                "name": name,
                "symbol": symbol,
                "value": round(current, 2),
                "change": round(current - prev, 2),
                "changePercent": f"{((current - prev) / prev * 100):.2f}%",
                "dayHigh": round(hist['High'].iloc[-1], 2),
                "dayLow": round(hist['Low'].iloc[-1], 2)
            })
        
        return json.dumps({
            "indices": index_data,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return f"Error getting market indices: {e}"


@yfinance_server.tool(
    name="get_etf_holdings",
    description="""Get top holdings information for an ETF.

Args:
    etf_ticker: str
        The ETF ticker symbol, e.g. "SPY", "QQQ", "VTI"
""",
)
async def get_etf_holdings(etf_ticker: str) -> str:
    """Get ETF holdings information"""
    try:
        etf = yf.Ticker(etf_ticker)
        info = etf.info
        
        result = {
            "ticker": etf_ticker,
            "name": info.get("shortName", "N/A"),
            "category": info.get("category", "N/A"),
            "totalAssets": f"${info.get('totalAssets', 0) / 1e9:.2f}B" if info.get("totalAssets") else "N/A",
            "expenseRatio": f"{info.get('annualReportExpenseRatio', 0) * 100:.2f}%" if info.get("annualReportExpenseRatio") else "N/A",
            "ytdReturn": f"{info.get('ytdReturn', 0) * 100:.2f}%" if info.get("ytdReturn") else "N/A",
            "threeYearReturn": f"{info.get('threeYearAverageReturn', 0) * 100:.2f}%" if info.get("threeYearAverageReturn") else "N/A",
            "fiveYearReturn": f"{info.get('fiveYearAverageReturn', 0) * 100:.2f}%" if info.get("fiveYearAverageReturn") else "N/A"
        }
        
        # Try to get holdings
        try:
            # This may not work for all ETFs
            if hasattr(etf, 'institutional_holders') and etf.institutional_holders is not None:
                result["topHolders"] = json.loads(etf.institutional_holders.head(10).to_json(orient="records"))
        except Exception:
            result["topHolders"] = "Holdings data not available"
        
        return json.dumps(result)
    except Exception as e:
        return f"Error getting ETF holdings for {etf_ticker}: {e}"


@yfinance_server.tool(
    name="compare_to_benchmark",
    description="""Compare a stock's performance to a benchmark index or ETF.

Args:
    ticker: str
        The stock ticker to compare
    benchmark: str
        The benchmark ticker. Default is "SPY" (S&P 500 ETF).
    period: str
        Comparison period. Default is "1y".
""",
)
async def compare_to_benchmark(ticker: str, benchmark: str = "SPY", period: str = "1y") -> str:
    """Compare stock to benchmark"""
    try:
        stock = get_cached_ticker(ticker)
        bench = get_cached_ticker(benchmark)
        
        stock_hist = stock.history(period=period)
        bench_hist = bench.history(period=period)
        
        if stock_hist.empty or bench_hist.empty:
            return "Error: Could not fetch data for comparison"
        
        # Calculate returns
        stock_start = stock_hist['Close'].iloc[0]
        stock_end = stock_hist['Close'].iloc[-1]
        stock_return = (stock_end - stock_start) / stock_start * 100
        
        bench_start = bench_hist['Close'].iloc[0]
        bench_end = bench_hist['Close'].iloc[-1]
        bench_return = (bench_end - bench_start) / bench_start * 100
        
        # Calculate volatility
        stock_vol = stock_hist['Close'].pct_change().std() * np.sqrt(252) * 100
        bench_vol = bench_hist['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # Calculate beta
        stock_returns = stock_hist['Close'].pct_change().dropna()
        bench_returns = bench_hist['Close'].pct_change().dropna()
        
        # Align the data
        min_len = min(len(stock_returns), len(bench_returns))
        stock_returns = stock_returns.iloc[-min_len:]
        bench_returns = bench_returns.iloc[-min_len:]
        
        covariance = stock_returns.cov(bench_returns)
        variance = bench_returns.var()
        beta = covariance / variance if variance != 0 else 1
        
        # Alpha (simplified Jensen's alpha)
        risk_free = 0.04  # Assume 4% risk-free rate
        alpha = stock_return - (risk_free + beta * (bench_return - risk_free))
        
        result = {
            "stock": {
                "ticker": ticker,
                "return": f"{stock_return:.2f}%",
                "volatility": f"{stock_vol:.2f}%"
            },
            "benchmark": {
                "ticker": benchmark,
                "return": f"{bench_return:.2f}%",
                "volatility": f"{bench_vol:.2f}%"
            },
            "comparison": {
                "outperformance": f"{(stock_return - bench_return):.2f}%",
                "beta": round(beta, 2),
                "alpha": f"{alpha:.2f}%",
                "relativeVolatility": f"{(stock_vol / bench_vol):.2f}x" if bench_vol > 0 else "N/A"
            },
            "period": period,
            "conclusion": "Outperformed" if stock_return > bench_return else "Underperformed"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error comparing to benchmark: {e}"


# ==================== ALERT & WATCHLIST FEATURES ====================

@yfinance_server.tool(
    name="check_price_targets",
    description="""Check if stocks in a watchlist have reached specified price targets.

Args:
    watchlist: dict
        Dictionary with tickers as keys and target prices as values, e.g. {"AAPL": 200, "GOOGL": 150}
""",
)
async def check_price_targets(watchlist: dict) -> str:
    """Check if stocks reached price targets"""
    try:
        alerts = []
        
        for ticker, target in watchlist.items():
            company = get_cached_ticker(ticker)
            info = company.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            
            status = "Target Reached" if current_price >= target else "Below Target"
            distance = (target - current_price) / current_price * 100
            
            alerts.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "currentPrice": round(current_price, 2),
                "targetPrice": target,
                "status": status,
                "distanceToTarget": f"{distance:.2f}%",
                "triggered": current_price >= target
            })
        
        # Separate triggered and not triggered
        triggered = [a for a in alerts if a["triggered"]]
        pending = [a for a in alerts if not a["triggered"]]
        
        return json.dumps({
            "triggered": triggered,
            "pending": pending,
            "totalWatched": len(alerts),
            "totalTriggered": len(triggered)
        })
    except Exception as e:
        return f"Error checking price targets: {e}"


@yfinance_server.tool(
    name="get_52_week_alerts",
    description="""Get alerts for stocks near their 52-week highs or lows.

Args:
    tickers: list[str]
        List of ticker symbols to check
    threshold: float
        Percentage threshold from high/low to trigger alert. Default is 5 (5%).
""",
)
async def get_52_week_alerts(tickers: list[str], threshold: float = 5) -> str:
    """Get 52-week high/low alerts"""
    try:
        alerts = []
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            info = company.info
            
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            week_52_high = info.get("fiftyTwoWeekHigh", 0)
            week_52_low = info.get("fiftyTwoWeekLow", 0)
            
            if current_price == 0 or week_52_high == 0:
                continue
            
            dist_from_high = (week_52_high - current_price) / week_52_high * 100
            dist_from_low = (current_price - week_52_low) / week_52_low * 100
            
            alert = {
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "currentPrice": round(current_price, 2),
                "52WeekHigh": round(week_52_high, 2),
                "52WeekLow": round(week_52_low, 2),
                "distanceFromHigh": f"{dist_from_high:.2f}%",
                "distanceFromLow": f"{dist_from_low:.2f}%"
            }
            
            if dist_from_high <= threshold:
                alert["alert"] = "Near 52-Week High"
                alert["alertType"] = "high"
                alerts.append(alert)
            elif dist_from_low <= threshold and week_52_low > 0:
                low_threshold = (current_price - week_52_low) / current_price * 100
                if low_threshold <= threshold:
                    alert["alert"] = "Near 52-Week Low"
                    alert["alertType"] = "low"
                    alerts.append(alert)
        
        # Separate by alert type
        near_highs = [a for a in alerts if a.get("alertType") == "high"]
        near_lows = [a for a in alerts if a.get("alertType") == "low"]
        
        return json.dumps({
            "nearHighs": near_highs,
            "nearLows": near_lows,
            "totalAlerts": len(alerts),
            "threshold": f"{threshold}%"
        })
    except Exception as e:
        return f"Error getting 52-week alerts: {e}"


@yfinance_server.tool(
    name="get_unusual_volume",
    description="""Detect stocks with unusual trading volume.

Args:
    tickers: list[str]
        List of ticker symbols to check
    threshold: float
        Volume ratio threshold to flag as unusual. Default is 2.0 (2x average volume).
""",
)
async def get_unusual_volume(tickers: list[str], threshold: float = 2.0) -> str:
    """Detect unusual volume"""
    try:
        unusual = []
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            hist = company.history(period="1mo")
            info = company.info
            
            if hist.empty:
                continue
            
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            if avg_volume == 0:
                continue
            
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio >= threshold:
                unusual.append({
                    "ticker": ticker,
                    "name": info.get("shortName", "N/A"),
                    "currentVolume": int(current_volume),
                    "averageVolume": int(avg_volume),
                    "volumeRatio": round(volume_ratio, 2),
                    "priceChange": f"{hist['Close'].pct_change().iloc[-1] * 100:.2f}%",
                    "signal": "High Volume Spike"
                })
        
        # Sort by volume ratio
        unusual.sort(key=lambda x: x["volumeRatio"], reverse=True)
        
        return json.dumps({
            "unusualVolume": unusual,
            "totalFlagged": len(unusual),
            "threshold": f"{threshold}x average"
        })
    except Exception as e:
        return f"Error detecting unusual volume: {e}"


@yfinance_server.tool(
    name="get_significant_movers",
    description="""Get stocks with significant price moves.

Args:
    tickers: list[str]
        List of ticker symbols to check
    threshold: float
        Minimum percentage move to be considered significant. Default is 3.0 (3%).
""",
)
async def get_significant_movers(tickers: list[str], threshold: float = 3.0) -> str:
    """Get significant price movers"""
    try:
        movers = []
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            hist = company.history(period="5d")
            info = company.info
            
            if hist.empty or len(hist) < 2:
                continue
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            pct_change = (current_price - prev_price) / prev_price * 100
            
            if abs(pct_change) >= threshold:
                movers.append({
                    "ticker": ticker,
                    "name": info.get("shortName", "N/A"),
                    "currentPrice": round(current_price, 2),
                    "previousClose": round(prev_price, 2),
                    "change": round(current_price - prev_price, 2),
                    "changePercent": f"{pct_change:.2f}%",
                    "direction": "Gainer" if pct_change > 0 else "Loser",
                    "volume": int(hist['Volume'].iloc[-1])
                })
        
        # Separate gainers and losers
        gainers = sorted([m for m in movers if m["direction"] == "Gainer"], 
                        key=lambda x: float(x["changePercent"].rstrip('%')), reverse=True)
        losers = sorted([m for m in movers if m["direction"] == "Loser"], 
                       key=lambda x: float(x["changePercent"].rstrip('%')))
        
        return json.dumps({
            "gainers": gainers,
            "losers": losers,
            "totalMovers": len(movers),
            "threshold": f"{threshold}%"
        })
    except Exception as e:
        return f"Error getting significant movers: {e}"


# ==================== HISTORICAL ANALYSIS TOOLS ====================

@yfinance_server.tool(
    name="calculate_returns",
    description="""Calculate returns for a stock over various periods.

Args:
    ticker: str
        The ticker symbol
    periods: list[str]
        List of periods to calculate. Options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd". Default includes common periods.
""",
)
async def calculate_returns(
    ticker: str, 
    periods: list[str] = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"]
) -> str:
    """Calculate returns over various periods"""
    try:
        company = get_cached_ticker(ticker)
        info = company.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        
        returns_data = {}
        
        for period in periods:
            hist = company.history(period=period)
            if hist.empty:
                returns_data[period] = "N/A"
                continue
            
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            period_return = (end_price - start_price) / start_price * 100
            returns_data[period] = f"{period_return:.2f}%"
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "currentPrice": round(current_price, 2),
            "returns": returns_data,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating returns for {ticker}: {e}"


@yfinance_server.tool(
    name="get_volatility_analysis",
    description="""Get volatility analysis for a stock.

Args:
    ticker: str
        The ticker symbol
    period: str
        Historical period to analyze. Default is "1y".
""",
)
async def get_volatility_analysis(ticker: str, period: str = "1y") -> str:
    """Get volatility analysis"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period=period)
        info = company.info
        
        if hist.empty:
            return f"Error: No data for {ticker}"
        
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate various volatility measures
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
        rolling_vol_60 = returns.rolling(window=60).std().iloc[-1] * np.sqrt(252)
        
        # Average True Range
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(window=14).mean().iloc[-1]
        atr_percent = atr / hist['Close'].iloc[-1] * 100
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "period": period,
            "volatility": {
                "dailyVolatility": f"{daily_vol * 100:.2f}%",
                "annualizedVolatility": f"{annual_vol * 100:.2f}%",
                "rolling20DayVol": f"{rolling_vol_20 * 100:.2f}%",
                "rolling60DayVol": f"{rolling_vol_60 * 100:.2f}%"
            },
            "atr": {
                "value": round(atr, 2),
                "percent": f"{atr_percent:.2f}%"
            },
            "priceRange": {
                "high": round(hist['High'].max(), 2),
                "low": round(hist['Low'].min(), 2),
                "range": round(hist['High'].max() - hist['Low'].min(), 2),
                "rangePercent": f"{((hist['High'].max() - hist['Low'].min()) / hist['Low'].min() * 100):.2f}%"
            },
            "beta": round(info.get("beta", 1), 2) if info.get("beta") else "N/A",
            "volatilityRating": "High" if annual_vol > 0.4 else "Medium" if annual_vol > 0.2 else "Low"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error analyzing volatility for {ticker}: {e}"


@yfinance_server.tool(
    name="get_max_drawdown",
    description="""Calculate maximum drawdown for a stock.

Args:
    ticker: str
        The ticker symbol
    period: str
        Historical period to analyze. Default is "1y".
""",
)
async def get_max_drawdown(ticker: str, period: str = "1y") -> str:
    """Calculate maximum drawdown"""
    try:
        company = get_cached_ticker(ticker)
        hist = company.history(period=period)
        info = company.info
        
        if hist.empty:
            return f"Error: No data for {ticker}"
        
        close = hist['Close']
        
        # Calculate running maximum
        running_max = close.expanding().max()
        
        # Calculate drawdown
        drawdown = (close - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_value = running_max.loc[max_dd_date]
        peak_date = close[close == peak_value].index[0] if not close[close == peak_value].empty else "N/A"
        
        # Current drawdown from all-time high
        current_dd = (close.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1] * 100
        
        # Recovery analysis
        recovery_days = None
        if max_dd_date != hist.index[-1]:
            post_dd = close[close.index > max_dd_date]
            recovery_point = post_dd[post_dd >= peak_value]
            if not recovery_point.empty:
                recovery_days = (recovery_point.index[0] - max_dd_date).days
        
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "period": period,
            "maxDrawdown": {
                "percentage": f"{max_dd:.2f}%",
                "peakDate": peak_date.strftime("%Y-%m-%d") if hasattr(peak_date, 'strftime') else str(peak_date),
                "troughDate": max_dd_date.strftime("%Y-%m-%d"),
                "peakValue": round(peak_value, 2),
                "troughValue": round(close.loc[max_dd_date], 2),
                "recoveryDays": recovery_days if recovery_days else "Not yet recovered"
            },
            "currentDrawdown": f"{current_dd:.2f}%",
            "periodHigh": round(close.max(), 2),
            "periodLow": round(close.min(), 2),
            "currentPrice": round(close.iloc[-1], 2)
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error calculating max drawdown for {ticker}: {e}"


@yfinance_server.tool(
    name="compare_performance",
    description="""Compare performance of multiple stocks over a period.

Args:
    tickers: list[str]
        List of ticker symbols to compare
    period: str
        Comparison period. Default is "1y".
""",
)
async def compare_performance(tickers: list[str], period: str = "1y") -> str:
    """Compare performance of multiple stocks"""
    try:
        performance_data = []
        
        for ticker in tickers:
            company = get_cached_ticker(ticker)
            hist = company.history(period=period)
            info = company.info
            
            if hist.empty:
                continue
            
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price * 100
            
            # Calculate additional metrics
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            sharpe = (total_return / 100 - 0.04) / volatility if volatility > 0 else 0
            
            # Max drawdown
            running_max = hist['Close'].expanding().max()
            drawdown = ((hist['Close'] - running_max) / running_max * 100).min()
            
            performance_data.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "startPrice": round(start_price, 2),
                "endPrice": round(end_price, 2),
                "totalReturn": f"{total_return:.2f}%",
                "annualizedVolatility": f"{volatility * 100:.2f}%",
                "sharpeRatio": round(sharpe, 2),
                "maxDrawdown": f"{drawdown:.2f}%",
                "52WeekHigh": round(hist['High'].max(), 2),
                "52WeekLow": round(hist['Low'].min(), 2)
            })
        
        # Sort by total return
        performance_data.sort(key=lambda x: float(x['totalReturn'].rstrip('%')), reverse=True)
        
        # Add rankings
        for i, data in enumerate(performance_data):
            data["rank"] = i + 1
        
        result = {
            "comparison": performance_data,
            "period": period,
            "bestPerformer": performance_data[0]["ticker"] if performance_data else "N/A",
            "worstPerformer": performance_data[-1]["ticker"] if performance_data else "N/A",
            "averageReturn": f"{sum(float(d['totalReturn'].rstrip('%')) for d in performance_data) / len(performance_data):.2f}%" if performance_data else "N/A"
        }
        
        return json.dumps(result)
    except Exception as e:
        return f"Error comparing performance: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting Yahoo Finance MCP server...")
    yfinance_server.run(transport="stdio")
