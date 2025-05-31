import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
import colorama
from colorama import Fore, Back, Style
import threading

# Initialize colorama (required for Windows)
colorama.init(autoreset=True)

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y-%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)


class PrettyFormatter(logging.Formatter):
    """Custom formatter for pretty colored logs with symbols"""

    # Define log format with colors and symbols
    FORMATS = {
        logging.DEBUG: f"{Fore.CYAN}[•] %(asctime)s {Fore.WHITE}| {Fore.CYAN}%(name)-15s {Fore.WHITE}| {Fore.CYAN}DEBUG{Style.RESET_ALL} | %(message)s",
        logging.INFO: f"{Fore.GREEN}[✓] %(asctime)s {Fore.WHITE}| {Fore.GREEN}%(name)-15s {Fore.WHITE}| {Fore.GREEN}INFO{Style.RESET_ALL}  | %(message)s",
        logging.WARNING: f"{Fore.YELLOW}[⚠] %(asctime)s {Fore.WHITE}| {Fore.YELLOW}%(name)-15s {Fore.WHITE}| {Fore.YELLOW}WARN{Style.RESET_ALL}  | %(message)s",
        logging.ERROR: f"{Fore.RED}[✗] %(asctime)s {Fore.WHITE}| {Fore.RED}%(name)-15s {Fore.WHITE}| {Fore.RED}ERROR{Style.RESET_ALL} | %(message)s",
        logging.CRITICAL: f"{Fore.MAGENTA}[!] %(asctime)s {Fore.WHITE}| {Fore.MAGENTA}%(name)-15s {Fore.WHITE}| {Fore.MAGENTA}CRIT{Style.RESET_ALL}  | %(message)s"
    }

    # Plain format for file logging (no colors)
    FILE_FORMATS = {
        logging.DEBUG: "[•] %(asctime)s | %(name)-15s | DEBUG | %(message)s",
        logging.INFO: "[✓] %(asctime)s | %(name)-15s | INFO  | %(message)s",
        logging.WARNING: "[⚠] %(asctime)s | %(name)-15s | WARN  | %(message)s",
        logging.ERROR: "[✗] %(asctime)s | %(name)-15s | ERROR | %(message)s",
        logging.CRITICAL: "[!] %(asctime)s | %(name)-15s | CRIT  | %(message)s"
    }

    def __init__(self, use_color=True):
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        # Select colored or plain format based on the destination
        log_format = self.FORMATS.get(record.levelno) if self.use_color else self.FILE_FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class SectionLogger:
    """Helper class to create section headers in logs"""

    @staticmethod
    def section(title, logger=None, level=logging.INFO, char='=', length=50):
        """Log a section header with decorative characters"""
        if logger is None:
            logger = logging.getLogger()

        # Create a decorative header
        header = f"\n{char * length}\n{title.center(length)}\n{char * length}"

        if level == logging.INFO:
            logger.info(f"{Fore.BLUE}{header}{Style.RESET_ALL}")
        elif level == logging.DEBUG:
            logger.debug(f"{Fore.CYAN}{header}{Style.RESET_ALL}")
        elif level == logging.WARNING:
            logger.warning(f"{Fore.YELLOW}{header}{Style.RESET_ALL}")
        elif level == logging.ERROR:
            logger.error(f"{Fore.RED}{header}{Style.RESET_ALL}")
        elif level == logging.CRITICAL:
            logger.critical(f"{Fore.MAGENTA}{header}{Style.RESET_ALL}")


# Thread-safe flag and lock for logger configuration
_logger_configured = False
_config_lock = threading.Lock()


def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler with pretty formatting.
    Thread-safe and only configures once to prevent duplicate handlers.
    """
    global _logger_configured

    # Thread-safe check to prevent multiple configurations
    with _config_lock:
        if _logger_configured:
            return

        # Get the root logger
        logger = logging.getLogger()

        # Only configure if no handlers exist or if they're not our custom handlers
        if not logger.handlers or not any(
                isinstance(h, (RotatingFileHandler, logging.StreamHandler)) for h in logger.handlers):
            # Set logger level
            logger.setLevel(logging.DEBUG)

            # Console handler with colored formatter
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(PrettyFormatter(use_color=True))
            console_handler.setLevel(logging.INFO)
            console_handler.set_name("console_handler")  # Name the handler for identification

            # File handler with plain formatter and rotation
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            file_handler.setFormatter(PrettyFormatter(use_color=False))
            file_handler.setLevel(logging.INFO)
            file_handler.set_name("file_handler")  # Name the handler for identification

            # Add handlers to the logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Log startup information
            logger.info(f"Logger initialized. Log file: {log_file_path}")

        # Set the flag to indicate logger is configured
        _logger_configured = True


def get_logger(name=None):
    """
    Get a logger instance with the specified name.
    Ensures the logger is properly configured.
    """
    configure_logger()
    return logging.getLogger(name)


# Configure the logger on import
configure_logger()

# Export the section logger for use in other modules
section = SectionLogger.section