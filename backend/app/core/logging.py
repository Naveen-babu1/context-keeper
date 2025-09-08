"""Logging configuration"""
import logging

def setup_logging(level: str = "INFO"):
    """Configure application logging"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]"
    )
