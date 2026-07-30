import pandas as pd

class DataLoader:
    """
    Skeleton class to handle loading, cleaning, and technical feature engineering
    for stock price data.
    """
    def __init__(self, data_dir: str):
        """
        Initializes the DataLoader with the path to the data directory.
        """
        self.data_dir = data_dir

    def get_available_tickers(self) -> list:
        """
        Returns a list of available stock tickers.
        """
        # TODO: Implement ticker scanning logic
        pass

    def load_data(self, ticker: str) -> pd.DataFrame:
        """
        Loads CSV data for a given ticker and returns a DataFrame.
        """
        # TODO: Implement data loading and cleaning logic
        pass

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and appends technical indicators to the DataFrame.
        """
        # TODO: Implement indicator calculations (RSI, MACD, Bollinger Bands, etc.)
        pass
