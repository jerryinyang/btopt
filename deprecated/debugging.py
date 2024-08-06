logger_main.debug(f"Processing timestamp: {timestamp}")
logger_main.debug(f"Data point keys: {list(data_point.keys())}")

for symbol in data_point:
    logger_main.debug(f"Timeframes for {symbol}: {list(data_point[symbol].keys())}")

for strategy_id, strategy in self._strategies.items():
    logger_main.debug(f"Processing strategy: {strategy_id}")
    for symbol in self._optimized_dataview.symbols:
        logger_main.debug(f"Processing symbol: {symbol}")
        logger_main.debug(f"Strategy primary timeframe: {strategy.primary_timeframe}")

        try:
            if self._optimized_dataview.is_original_data_point(
                symbol, strategy.primary_timeframe, timestamp
            ):
                logger_main.debug(
                    f"Original data point found for {symbol} at {strategy.primary_timeframe}"
                )
                ohlcv_data = data_point[symbol][strategy.primary_timeframe]
                logger_main.debug(
                    f"OHLCV data for {symbol} at {strategy.primary_timeframe}: {ohlcv_data}"
                )
                bar = self._create_bar(symbol, strategy.primary_timeframe, ohlcv_data)
                strategy.process_bar(bar)
            else:
                logger_main.debug(
                    f"No original data point for {symbol} at {strategy.primary_timeframe}"
                )
        except KeyError as e:
            logger_main.warning(
                f"Missing data for {symbol} at {strategy.primary_timeframe}: {e}"
            )
            logger_main.debug(
                f"Available timeframes for {symbol}: {list(data_point[symbol].keys()) if symbol in data_point else 'N/A'}"
            )
            continue


# In run()

logger_main.debug(f"Available timeframes: {self._optimized_dataview.timeframes}")
# Check if data is available for all strategies
for strategy_id, strategy in self._strategies.items():
    logger_main.debug(
        f"Strategy {strategy_id} primary timeframe: {strategy.primary_timeframe}"
    )
    if strategy.primary_timeframe not in self._optimized_dataview.timeframes:
        logger_main.log_and_raise(
            ValueError(
                f"Data for primary timeframe {strategy.primary_timeframe} (type: {type(strategy.primary_timeframe)})is not available for strategy {strategy_id}. Available timeframes: {self._optimized_dataview.timeframes}"
            )
        )
