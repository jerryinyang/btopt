from abc import ABC, abstractmethod
from decimal import ROUND_DOWN
from typing import Any, Dict, Optional, Union

from .parameters import Parameters
from .types import StrategyType
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


class Sizer(ABC):
    def __init__(self, params: Optional[Parameters] = None):
        if params:
            if not isinstance(params, Parameters):
                logger_main.log_and_raise(
                    TypeError("params must be an instance of Parameters")
                )
            self.params = params
        else:
            self.params = Parameters()

    @abstractmethod
    def calculate_position_size(
        self,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
        risk_amount: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Calculate the position size based on the given parameters.

        Args:
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade (e.g., stop loss).
            risk_amount (ExtendedDecimal): The amount in cash to risk on the trade

        Returns:
            ExtendedDecimal: The calculated position size.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        pass

    @abstractmethod
    def validate_inputs(
        self,
        risk_amount: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
    ) -> None:
        """
        Validate the inputs for the position size calculation.

        Args:
            risk_amount (ExtendedDecimal): The amount of risk to take on the trade.
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        pass

    def update_params(self, new_params: Union[Parameters, Dict[str, Any]]) -> None:
        """
        Update the parameters of the sizer.

        Args:
            new_params (Union[Parameters, Dict[str, Any]]): New parameters to update or add.

        Raises:
            TypeError: If new_params is neither a Parameters object nor a dictionary.
        """
        self.params.update(new_params)

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the position sizer.

        Returns:
            Dict[str, Any]: A dictionary containing information about the position sizer.
        """
        pass


class NaiveSizer(Sizer):
    def __init__(self, params: Optional[Parameters] = None):
        super().__init__(params)
        self.min_position_size = self.params.get(
            "min_position_size", ExtendedDecimal("0.00001")
        )
        self.max_position_size = self.params.get(
            "max_position_size", ExtendedDecimal("100000")
        )

    def calculate_position_size(
        self,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
        risk_amount: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Calculate the position size based on the strategy's risk parameters and entry price.
        If exit_price is provided, it uses (entry_price - exit_price) as the risk per unit.
        If exit_price is not provided, it uses entry_price as the risk per unit.

        Args:
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade (e.g., stop loss).
            risk_amount (ExtendedDecimal): The amount in cash to risk on the trade

        Returns:
            ExtendedDecimal: The calculated position size.

        Raises:
            ValueError: If the input parameters are invalid or the strategy is not properly initialized.
        """

        # Validate the calculation parameters
        self.validate_inputs(risk_amount, entry_price, exit_price)

        if exit_price is not None:
            risk_per_unit = abs(entry_price - exit_price)
        else:
            risk_per_unit = entry_price

        if risk_per_unit == ExtendedDecimal("0"):
            logger_main.log_and_raise(ValueError("Risk per unit cannot be zero."))

        # Calculate the position size
        position_size = risk_amount / risk_per_unit

        logger_main.warning(
            f"<<< CALCULATING POSITION SIZE >>>\nRISK AMOUNT: {risk_amount}\nRISK PER UNIT: {risk_per_unit}\nPOSITION SIZE: {position_size}\n"
        )

        # Apply min and max position size constraints
        position_size = max(
            min(position_size, self.max_position_size), self.min_position_size
        )

        return position_size.quantize(
            ExtendedDecimal("0.00001"), rounding=ROUND_DOWN
        )  # Round to fractional units

    def validate_inputs(
        self,
        risk_amount: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
    ) -> None:
        """
        Validate the inputs for the position size calculation.

        Args:
            risk_amount (ExtendedDecimal): The amount of risk to take on the trade.
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if risk_amount <= ExtendedDecimal("0"):
            logger_main.log_and_raise(
                ValueError(
                    "Invalid risk amount: Risk amount must be a positive decimal, greater than 0"
                )
            )

        if entry_price <= ExtendedDecimal("0"):
            logger_main.log_and_raise(
                ValueError("Invalid entry price: Entry price must be positive")
            )

        if exit_price is not None:
            if exit_price <= ExtendedDecimal("0"):
                logger_main.log_and_raise(
                    ValueError("Invalid exit price: Exit price must be positive")
                )

            if exit_price == entry_price:
                logger_main.warning("Exit price is equal to entry price")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the position sizer.

        Returns:
            Dict[str, Any]: A dictionary containing information about the position sizer.
        """
        return {
            "name": "NaivePositionSizer",
            "description": "A simple position sizer based on risk amount and entry price",
            "min_position_size": str(self.min_position_size),
            "max_position_size": str(self.max_position_size),
        }


class ForexSizer(Sizer):
    def __init__(self, params: Optional[Parameters] = None):
        super().__init__(params)
        self.min_lot_size = self.params.get("min_lot_size", ExtendedDecimal("0.01"))
        self.max_lot_size = self.params.get("max_lot_size", ExtendedDecimal("100"))
        self.pip_value = self.params.get("pip_value", ExtendedDecimal("0.0001"))
        self.account_currency = self.params.get("account_currency", "USD")
        self.units_per_lot = ExtendedDecimal("100000")  # Standard lot size

    def calculate_position_size(
        self,
        strategy: StrategyType,
        symbol: str,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
    ) -> ExtendedDecimal:
        """
        Calculate the position size for a Forex trade based on the strategy's risk parameters and entry price.

        Args:
            strategy (StrategyType): The strategy object, providing context and risk parameters.
            symbol (str): The Forex pair symbol (e.g., "EURUSD").
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade (e.g., stop loss).

        Returns:
            ExtendedDecimal: The calculated position size in units (not lots).

        Raises:
            ValueError: If the input parameters are invalid or the strategy is not properly initialized.
        """
        if not strategy or not isinstance(strategy, StrategyType):
            logger_main.log_and_raise(ValueError("Invalid strategy object provided."))

        engine = strategy._engine
        if not engine:
            logger_main.log_and_raise(ValueError("Strategy engine is not initialized."))

        risk_percentage = strategy.risk_percentage

        # Validate the calculation parameters
        self.validate_inputs(risk_percentage, entry_price, exit_price)

        # Get the risk amount for the symbol from the Engine
        account_balance = engine.get_account_balance(strategy._id)
        risk_amount = account_balance * risk_percentage

        # Calculate pip difference
        if exit_price is not None:
            pip_difference = abs(entry_price - exit_price) / self.pip_value
        else:
            pip_difference = ExtendedDecimal(
                "1"
            )  # Default to 1 pip if no exit price is provided

        # Calculate pip value in account currency
        pip_value_account_currency = self.calculate_pip_value(symbol, engine)

        # Calculate position size in lots
        position_size_lots = risk_amount / (pip_difference * pip_value_account_currency)

        # Convert position size from lots to units
        position_size_units = position_size_lots * self.units_per_lot

        # Apply min and max lot size constraints (converting to units)
        min_units = self.min_lot_size * self.units_per_lot
        max_units = self.max_lot_size * self.units_per_lot
        position_size_units = max(min(position_size_units, max_units), min_units)

        return position_size_units.quantize(
            ExtendedDecimal("1")
        )  # Round to whole units

    def validate_inputs(
        self,
        risk_percentage: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal],
    ) -> None:
        """
        Validate the inputs for the position size calculation.

        Args:
            risk_percentage (ExtendedDecimal): The percentage of account balance to risk on the trade.
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade.

        Raises:
            ValueError: If any of the inputs are invalid.
        """
        if risk_percentage <= ExtendedDecimal("0") or risk_percentage > ExtendedDecimal(
            "1"
        ):
            logger_main.log_and_raise(
                ValueError(
                    "Invalid risk percentage: Risk percentage must be a positive decimal between 0 and 1"
                )
            )

        if entry_price <= ExtendedDecimal("0"):
            logger_main.log_and_raise(
                ValueError("Invalid entry price: Entry price must be positive")
            )

        if exit_price is not None:
            if exit_price <= ExtendedDecimal("0"):
                logger_main.log_and_raise(
                    ValueError("Invalid exit price: Exit price must be positive")
                )

            if exit_price == entry_price:
                logger_main.warning("Exit price is equal to entry price")

    def calculate_pip_value(self, symbol: str, engine: Any) -> ExtendedDecimal:
        """
        Calculate the pip value for a given Forex pair in the account currency.

        Args:
            symbol (str): The Forex pair symbol (e.g., "EURUSD").
            engine (Any): The trading engine, used to get current market rates.

        Returns:
            ExtendedDecimal: The calculated pip value in the account currency.

        Raises:
            ValueError: If the symbol is invalid or required rates are not available.
        """
        base_currency = symbol[:3]
        quote_currency = symbol[3:]

        if quote_currency == self.account_currency:
            return self.pip_value
        elif base_currency == self.account_currency:
            current_rate = engine.get_current_rate(symbol)
            return self.pip_value / current_rate
        else:
            # Need to perform two conversions
            current_rate = engine.get_current_rate(symbol)
            conversion_rate = engine.get_current_rate(
                f"{quote_currency}{self.account_currency}"
            )
            return (self.pip_value / current_rate) * conversion_rate

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the Forex position sizer.

        Returns:
            Dict[str, Any]: A dictionary containing information about the position sizer.
        """
        return {
            "name": "ForexPositionSizer",
            "description": "A position sizer specifically designed for Forex trading",
            "min_lot_size": str(self.min_lot_size),
            "max_lot_size": str(self.max_lot_size),
            "pip_value": str(self.pip_value),
            "account_currency": self.account_currency,
        }
