
import os
import logging
from backend.app.ops import pathmap, config

logger = logging.getLogger(__name__)

def ensure_dirs() -> None:
    """
    Creates all root-level directories defined in pathmap.
    """
    paths = pathmap.get_paths()
    
    # List of dirs to create
    dirs = [
        paths.data_raw,
        paths.data_canonical,
        paths.features,
        paths.priors,
        paths.datasets,
        paths.models,
        paths.calibration,
        paths.manifests,
        paths.outputs,
        paths.logs,
        # Legacy
        paths.legacy_artifacts_root
    ]
    
    # Subdirs implied by pathmap?
    # e.g. data_canonical/ohlcv_adj is created dynamically by ingest
    # But we should ensure roots.
    
    for d in dirs:
        if not os.path.exists(d):
            logger.info(f"Creating directory: {d}")
            os.makedirs(d, exist_ok=True)
        # Check writable?
        assert_writable(d)
        
    logger.info("All operational directories verified.")

def assert_writable(path: str) -> None:
    """
    Checks if a path is writable.
    """
    if not os.access(path, os.W_OK):
        # Try to creat a test file
        try:
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except Exception as e:
            if config.FAIL_ON_MISSING_DIRS:
                raise PermissionError(f"Directory {path} is not writable: {e}")
            else:
                logger.warning(f"Directory {path} is not writable: {e}")
