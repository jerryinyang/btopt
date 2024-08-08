from typing import Any, Dict, List, Optional, Type

from .log_config import logger_main


class Outputs:
    """
    A class to manage indicator outputs with restrictions on setting values.

    This class is similar to the Parameters class but is specifically designed for
    indicator outputs. It allows setting values only at index [0] and provides
    error checking to prevent setting values at other indices.

    Attributes:
        _outputs (Dict[str, List[Any]]): Internal dictionary to store output names and their values.
        _types (Dict[str, Type]): Dictionary to store output names and their expected types.

    Methods:
        __getattr__: Allows dot notation access to outputs.
        __setattr__: Allows setting outputs with dot notation and performs type checking.
        __repr__: Returns a string representation of the outputs.
        set: Set an output value with optional type specification.
        get: Get an output value with an optional default.
        validate: Validate all outputs against their specified types.
    """

    def __init__(self, initial_outputs: Optional[Dict[str, Any]] = None):
        """
        Initialize the Outputs instance.

        Args:
            initial_outputs (Optional[Dict[str, Any]]): Initial outputs to set. Defaults to None.
        """
        self._outputs: Dict[str, List[Any]] = {}
        self._types: Dict[str, Type] = {}

        if initial_outputs:
            for key, value in initial_outputs.items():
                self.set(key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Allow dot notation access to outputs.

        Args:
            name (str): The name of the output to access.

        Returns:
            Any: The value of the requested output.

        Raises:
            AttributeError: If the output does not exist.
        """
        if name in self._outputs:
            return self._outputs[name][0]
        logger_main.log_and_raise(AttributeError(f"Output '{name}' does not exist"))

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Allow setting outputs with dot notation and perform type checking.

        Args:
            name (str): The name of the output to set.
            value (Any): The value to set for the output.

        Raises:
            AttributeError: If trying to set a non-existent output.
            ValueError: If trying to set a value at an index other than 0.
            TypeError: If the value type does not match the expected type.
        """
        if name.startswith("_"):
            # Allow setting private attributes
            super().__setattr__(name, value)
        elif name in self._outputs:
            self.set(name, value)
        else:
            logger_main.log_and_raise(
                AttributeError(
                    f"Cannot set new output '{name}' using dot notation. Use set() method instead."
                )
            )

    def __repr__(self) -> str:
        """
        Return a string representation of the outputs.

        Returns:
            str: A string representation of the outputs.
        """
        return f"Outputs({self._outputs})"

    def set(self, name: str, value: Any, output_type: Optional[Type] = None) -> None:
        """
        Set an output value with optional type specification.

        Args:
            name (str): The name of the output to set.
            value (Any): The value to set for the output.
            output_type (Optional[Type]): The expected type of the output. If None, infer from value.

        Raises:
            ValueError: If trying to set a value at an index other than 0.
            TypeError: If the value type does not match the expected type.
        """
        if output_type is None:
            output_type = type(value)

        if name in self._types and not isinstance(value, self._types[name]):
            logger_main.log_and_raise(
                TypeError(
                    f"Output '{name}' must be of type {self._types[name].__name__}"
                )
            )

        if name not in self._outputs:
            self._outputs[name] = [value]
            self._types[name] = output_type
        else:
            if len(self._outputs[name]) != 1:
                logger_main.log_and_raise(
                    ValueError(
                        f"Cannot set value for output '{name}' at index other than 0"
                    )
                )
            self._outputs[name][0] = value

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get an output value with an optional default.

        Args:
            name (str): The name of the output to get.
            default (Any): The default value to return if the output doesn't exist.

        Returns:
            Any: The value of the output or the default value.
        """
        if name in self._outputs:
            return self._outputs[name][0]
        return default

    def validate(self) -> bool:
        """
        Validate all outputs against their specified types.

        Returns:
            bool: True if all outputs are valid, False otherwise.

        Raises:
            TypeError: If any output value does not match its expected type.
        """
        for name, values in self._outputs.items():
            if not isinstance(values[0], self._types[name]):
                logger_main.log_and_raise(
                    TypeError(
                        f"Output '{name}' must be of type {self._types[name].__name__}"
                    )
                )
        return True

    def __iter__(self):
        """
        Allow iteration over output items.

        Yields:
            Tuple[str, Any]: A tuple containing the output name and value.
        """
        return iter((name, values[0]) for name, values in self._outputs.items())

    def keys(self):
        """
        Return a view of output names.

        Returns:
            Dict_keys: A view object of output names.
        """
        return self._outputs.keys()

    def values(self):
        """
        Return a view of output values.

        Returns:
            List[Any]: A list of the first (and only) value for each output.
        """
        return [values[0] for values in self._outputs.values()]

    def items(self):
        """
        Return a view of output items.

        Returns:
            List[Tuple[str, Any]]: A list of tuples containing output names and their first (and only) values.
        """
        return [(name, values[0]) for name, values in self._outputs.items()]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert outputs to a regular dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the outputs.
        """
        return {name: values[0] for name, values in self._outputs.items()}
