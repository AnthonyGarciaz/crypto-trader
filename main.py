import os
import time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import logging
import asyncio
import random
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpaca API Key and Secret from environment
api_key = os.getenv('APCA_API_KEY_ID')
api_secret = os.getenv('APCA_API_SECRET_KEY')

# Debugging output to check if the environment variables are loaded correctly
logger.info(f"API Key: {api_key}")
logger.info(f"API Secret: {api_secret}")

if not api_key or not api_secret:
    raise ValueError("API credentials not found in environment variables.")

class AlpacaAPI:
    """Handles API connection and interactions with Alpaca."""

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url='https://paper-api.alpaca.markets')

    def validate_connection(self):
        """Validate API connection and account status."""
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):.2f}")
            return account
        except tradeapi.rest.APIError as e:
            logger.error(f"API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_historical_data(self, symbol, timeframe=TimeFrame.Hour, limit=100, retries=3, delay=5):
        """Fetch historical data for a symbol with retry logic."""
        attempt = 0
        while attempt < retries:
            try:
                end = datetime.now().replace(microsecond=0)
                start = end - timedelta(hours=limit * 2)
                start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')

                bars = self.api.get_crypto_bars(symbol, timeframe, start=start_str, end=end_str).df

                if bars.empty:
                    logger.warning(f"No data received for {symbol}")
                    return None

                return bars.tail(limit).sort_index()
            except tradeapi.rest.APIError as e:
                logger.error(f"API error while fetching data for {symbol}: {e}")
                attempt += 1
                if attempt < retries:
                    logger.info(f"Retrying data fetch for {symbol} ({attempt}/{retries})")
                    time.sleep(random.uniform(1, delay))
                else:
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None

class TradingSignals:
    """Handles the logic for calculating buy/sell signals based on SMAs."""

    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def calculate(self, data):
        """Calculate the trading signals using a simple SMA crossover strategy."""
        if data is None or len(data) < self.long_window:
            logger.warning("Insufficient data for signal calculation.")
            return False, False

        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()

        # Get the last two data points
        current_cross = data['SMA_short'].iloc[-1] > data['SMA_long'].iloc[-1]
        prev_cross = data['SMA_short'].iloc[-2] > data['SMA_long'].iloc[-2]

        buy_signal = not prev_cross and current_cross
        sell_signal = prev_cross and not current_cross

        if buy_signal or sell_signal:
            logger.info(f"Signal generated: {'BUY' if buy_signal else 'SELL'}")
            logger.info(f"Current price: ${data['close'].iloc[-1]:.2f}")
            logger.info(f"Short SMA: ${data['SMA_short'].iloc[-1]:.2f}")
            logger.info(f"Long SMA: ${data['SMA_long'].iloc[-1]:.2f}")

        return buy_signal, sell_signal

class OrderExecution:
    """Handles order placement and logging of orders."""

    def __init__(self, api, fee_rate=0.25):
        self.api = api
        self.fee_rate = fee_rate

    def place_order(self, symbol, qty, side, order_type="market"):
        """Place an order with fee calculation."""
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            current_price = latest_trade.price

            # Calculate estimated fee
            estimated_fee = qty * current_price * (self.fee_rate / 100)
            logger.info(f"Estimated fee for {qty} {symbol}: {estimated_fee:.6f}")

            order = self.api.submit_order(
                symbol=symbol, qty=qty, side=side, type=order_type, time_in_force='gtc'
            )
            logger.info(f"Order placed: {side} {qty} of {symbol} at ~${current_price:.2f}")
            return order
        except tradeapi.rest.APIError as e:
            logger.error(f"API error while placing order: {e}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")

class AlpacaTradingBot:
    """Main trading bot class that coordinates fetching data, calculating signals, and placing orders."""

    def __init__(self, symbols, api_key, api_secret, short_window=20, long_window=50, fee_rate=0.25):
        self.symbols = symbols
        self.api = AlpacaAPI(api_key, api_secret)
        self.signals = TradingSignals(short_window, long_window)
        self.order_execution = OrderExecution(self.api.api, fee_rate)

        self.api.validate_connection()

    async def run(self, check_interval=900):
        """Main strategy execution loop."""
        logger.info(f"Starting trading bot for {', '.join(self.symbols)}")

        while True:
            try:
                # Check if market is open
                clock = self.api.api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Market is closed. Next opening at {next_open}")
                    await asyncio.sleep(min(check_interval, (clock.next_open - clock.timestamp).seconds))

                # Fetch and process data concurrently for each symbol
                tasks = [self.process_symbol(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)

                # Wait for the next cycle
                await asyncio.sleep(check_interval)
            except KeyboardInterrupt:
                logger.info("Stopping trading bot...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes

    async def process_symbol(self, symbol):
        """Process each symbol's data, calculate signals, and place orders."""
        try:
            data = self.api.get_historical_data(symbol)
            if data is None:
                logger.error(f"Failed to fetch data for {symbol}, retrying in 5 minutes.")
                await asyncio.sleep(300)  # Retry after 5 minutes
                return

            # Calculate trading signals
            buy_signal, sell_signal = self.signals.calculate(data)

            # Execute trades based on signals
            if buy_signal:
                logger.info(f"Buy signal detected for {symbol}")
                self.order_execution.place_order(symbol, qty=1, side='buy')
            elif sell_signal:
                logger.info(f"Sell signal detected for {symbol}")
                self.order_execution.place_order(symbol, qty=1, side='sell')
            else:
                logger.info(f"No trading signals for {symbol}")
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")

if __name__ == "__main__":
    try:
        # Create and run the trading bot for BTC/USD and ETH/USD
        bot = AlpacaTradingBot(symbols=["BTC/USD", "ETH/USD"], api_key=api_key, api_secret=api_secret)
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
