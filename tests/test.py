from btopt.data.dataloader import CSVDataLoader
from btopt.engine import Engine


def main():
    # Load data for multiple symbols and timeframes
    symbols = ["MAUD", "MEUR", "EURUSD"]
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Initialize CSVDataLoader
    csv_loader = CSVDataLoader(
        symbols,
        "1m",
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    print(csv_loader.tickers)

    engine = Engine()
    engine.add_data(csv_loader)


if __name__ == "__main__":
    main()
