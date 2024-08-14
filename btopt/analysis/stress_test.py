import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StressTest(ABC):
    """
    Abstract base class for all stress tests.
    """

    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def run(self) -> None:
        """
        Run the stress test. This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def visualize(self) -> None:
        """
        Visualize the results of the stress test. This method should be implemented by subclasses.
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """
        Return the results of the stress test.
        """
        return self.results

    def _log_start(self) -> None:
        """
        Log the start of the stress test.
        """
        logger.info(f"Starting stress test: {self.name}")

    def _log_end(self) -> None:
        """
        Log the end of the stress test.
        """
        logger.info(f"Completed stress test: {self.name}")

    def _log_error(self, error: Exception) -> None:
        """
        Log any errors that occur during the stress test.
        """
        logger.error(f"Error in stress test {self.name}: {str(error)}")


class StressTestSuite:
    """
    A class to manage and run multiple stress tests.
    """

    def __init__(self, name: str):
        self.name = name
        self.tests: List[StressTest] = []

    def add_test(self, test: StressTest) -> None:
        """
        Add a stress test to the suite.

        Args:
            test (StressTest): The stress test to add.
        """
        self.tests.append(test)
        logger.info(f"Added test '{test.name}' to suite '{self.name}'")

    def run_all(self) -> None:
        """
        Run all stress tests in the suite.
        """
        logger.info(f"Running all tests in suite '{self.name}'")
        for test in self.tests:
            try:
                test.run()
            except Exception as e:
                test._log_error(e)

    def visualize_all(self) -> None:
        """
        Visualize results for all stress tests in the suite.
        """
        logger.info(f"Visualizing results for all tests in suite '{self.name}'")
        for test in self.tests:
            try:
                test.visualize()
            except Exception as e:
                test._log_error(e)

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get results from all stress tests in the suite.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping test names to their results.
        """
        return {test.name: test.get_results() for test in self.tests}

    def summary(self) -> str:
        """
        Generate a summary of the stress test suite.

        Returns:
            str: A summary string describing the suite and its tests.
        """
        summary = f"Stress Test Suite: {self.name}\n"
        summary += f"Number of tests: {len(self.tests)}\n"
        summary += "Tests:\n"
        for test in self.tests:
            summary += f"- {test.name}\n"
        return summary


# Example usage
if __name__ == "__main__":
    # This is just to demonstrate how these classes might be used
    class DummyTest(StressTest):
        def run(self):
            self._log_start()
            # Simulate some work
            self.results = {"dummy_metric": 42}
            self._log_end()

        def visualize(self):
            print(f"Visualizing {self.name}: {self.results}")

    # Create a test suite
    suite = StressTestSuite("Example Suite")

    # Add some dummy tests
    suite.add_test(DummyTest("Test 1"))
    suite.add_test(DummyTest("Test 2"))

    # Run all tests
    suite.run_all()

    # Visualize results
    suite.visualize_all()

    # Get all results
    all_results = suite.get_all_results()
    print(all_results)

    # Print summary
    print(suite.summary())
