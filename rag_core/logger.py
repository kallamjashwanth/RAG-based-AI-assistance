import logging

_logger = None

def setup_logger():
    global _logger
    if _logger:
        return _logger

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    _logger = logging.getLogger("rag")
    return _logger