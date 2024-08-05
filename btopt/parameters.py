from typing import Any, Dict, Optional, Type


class Parameters:
    """
    A class to manage strategy parameters with dot notation access and type validation.

    This class allows for easy access to parameters using dot notation (e.g., params.moving_average_period)
    while providing type checking and validation. It's designed to be used within trading strategies
    to manage configuration parameters efficiently and safely.

    Attributes:
        _params (Dict[str, Any]): Internal dictionary to store parameter names and values.
        _types (Dict[str, Type]): Dictionary to store parameter names and their expected types.

    Methods:
        __getattr__: Allows dot notation access to parameters.
        __setattr__: Allows setting parameters with dot notation and performs type checking.
        __repr__: Returns a string representation of the parameters.
        set: Set a parameter value with optional type specification.
        get: Get a parameter value with an optional default.
        validate: Validate all parameters against their specified types.
    """

    def __init__(self, initial_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Parameters instance.

        Args:
            initial_params (Optional[Dict[str, Any]]): Initial parameters to set. Defaults to None.
        """
        self._params: Dict[str, Any] = {}
        self._types: Dict[str, Type] = {}

        if initial_params:
            for key, value in initial_params.items():
                self.set(key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Allow dot notation access to parameters.

        Args:
            name (str): The name of the parameter to access.

        Returns:
            Any: The value of the requested parameter.

        Raises:
            AttributeError: If the parameter does not exist.
        """
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"Parameter '{name}' does not exist")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow setting parameters with dot notation and perform type checking.

        Args:
            name (str): The name of the parameter to set.
            value (Any): The value to set for the parameter.

        Raises:
            AttributeError: If trying to set a non-existent parameter.
            TypeError: If the value type does not match the expected type.
        """
        if name.startswith("_"):
            # Allow setting private attributes
            super().__setattr__(name, value)
        elif name in self._params:
            self.set(name, value)
        else:
            raise AttributeError(
                f"Cannot set new parameter '{name}' using dot notation. Use set() method instead."
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the parameters.

        Returns:
            str: A string representation of the parameters.
        """
        return f"Parameters({self._params})"

    def set(self, name: str, value: Any, param_type: Optional[Type] = None) -> None:
        """
        Set a parameter value with optional type specification.

        Args:
            name (str): The name of the parameter to set.
            value (Any): The value to set for the parameter.
            param_type (Optional[Type]): The expected type of the parameter. If None, infer from value.

        Raises:
            TypeError: If the value type does not match the expected type.
        """
        if param_type is None:
            param_type = type(value)

        if name in self._types and not isinstance(value, self._types[name]):
            raise TypeError(
                f"Parameter '{name}' must be of type {self._types[name].__name__}"
            )

        self._params[name] = value
        self._types[name] = param_type

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value with an optional default.

        Args:
            name (str): The name of the parameter to get.
            default (Any): The default value to return if the parameter doesn't exist.

        Returns:
            Any: The value of the parameter or the default value.
        """
        return self._params.get(name, default)

    def validate(self) -> bool:
        """
        Validate all parameters against their specified types.

        Returns:
            bool: True if all parameters are valid, False otherwise.

        Raises:
            TypeError: If any parameter value does not match its expected type.
        """
        for name, value in self._params.items():
            if not isinstance(value, self._types[name]):
                raise TypeError(
                    f"Parameter '{name}' must be of type {self._types[name].__name__}"
                )
        return True

    def __iter__(self):
        """
        Allow iteration over parameter items.

        Yields:
            Tuple[str, Any]: A tuple containing the parameter name and value.
        """
        return iter(self._params.items())

    def keys(self):
        """
        Return a view of parameter names.

        Returns:
            Dict_keys: A view object of parameter names.
        """
        return self._params.keys()

    def values(self):
        """
        Return a view of parameter values.

        Returns:
            Dict_values: A view object of parameter values.
        """
        return self._params.values()

    def items(self):
        """
        Return a view of parameter items.

        Returns:
            Dict_items: A view object of parameter items (name-value pairs).
        """
        return self._params.items()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a regular dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the parameters.
        """
        return self._params.copy()
