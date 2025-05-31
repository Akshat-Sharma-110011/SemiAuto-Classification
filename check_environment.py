# check_environment.py
import numpy as np
import cloudpickle


def verify_environment():
    print("Environment Verification Report")
    print("=" * 40)
    print(f"NumPy Version: {np.__version__}")
    print(f"NumPy Path: {np.__file__}")
    print(f"Cloudpickle Version: {cloudpickle.__version__}")

    try:
        np.show_config()
        print("NumPy test PASSED")
    except Exception as e:
        print(f"NumPy test FAILED: {str(e)}")


if __name__ == "__main__":
    verify_environment()