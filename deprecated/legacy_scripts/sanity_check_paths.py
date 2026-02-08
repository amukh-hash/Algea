
import logging
import sys
import os

# Ensure backend can be imported from root
sys.path.append(os.getcwd())

from backend.app.ops import bootstrap, pathmap, config

logging.basicConfig(level=logging.INFO)

def main():
    print("Running PR-1 Sanity Check...")
    
    # 1. Check Config
    print(f"Config: WRITE_BOTH_PATHS={config.WRITE_BOTH_PATHS}")
    print(f"Config: FAIL_ON_MISSING_DIRS={config.FAIL_ON_MISSING_DIRS}")
    
    # 2. Run Bootstrap
    print("Running ensure_dirs()...")
    bootstrap.ensure_dirs()
    
    # 3. Verify on Disk
    paths = pathmap.get_paths()
    roots = [
        paths.data_raw, paths.data_canonical, paths.features,
        paths.priors, paths.datasets, paths.models, paths.calibration,
        paths.manifests, paths.outputs, paths.logs
    ]
    
    all_ok = True
    for r in roots:
        if not os.path.isdir(r):
            print(f"FAIL: {r} does not exist!")
            all_ok = False
        else:
            print(f"OK: {r}")
            
    if all_ok:
        print("\nSUCCESS: All directories bootstrapped.")
    else:
        print("\nFAILURE: Some directories missing.")
        sys.exit(1)

if __name__ == "__main__":
    main()
