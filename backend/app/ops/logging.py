
import logging
import json
import sys

# Structured Logger
class StructuredLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None: extra = {}
        # Could enrich with run_id etc
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

logging.setLoggerClass(StructuredLogger)

def get_logger(name: str):
    return logging.getLogger(name)
