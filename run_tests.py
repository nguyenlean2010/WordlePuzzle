import unittest
import coverage

# Initialize coverage
cov = coverage.Coverage()
cov.start()

# Run tests
loader = unittest.TestLoader()
suite = loader.discover('tests')  # Discover all tests in the tests directory
runner = unittest.TextTestRunner()
runner.run(suite)

# Stop coverage and generate report
cov.stop()
cov.save()
cov.html_report()