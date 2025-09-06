"""
conftest.py: Master configuration for the pytest testing suite.

This file is automatically discovered by pytest and is used to define project-wide
fixtures, hooks, and plugins. It's the most robust place to configure the testing
environment programmatically.

In this case, we are using it to manually and explicitly load the environment
variables from our .env file at the very start of the test session, before any
tests are collected. This solves a tricky loading issue with the pytest-dotenv
plugin where the configuration was not being read early enough.
"""

from dotenv import load_dotenv


def pytest_sessionstart(session):
    """
    Called by pytest at the very beginning of a test session.
    This hook is used to programmatically load our .env file.
    """
    load_dotenv()
