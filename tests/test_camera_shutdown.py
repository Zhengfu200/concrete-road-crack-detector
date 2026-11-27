import time

import pytest

from inference import LatestCamera


def blocked_driver(index,frames,errors,stopped):
    # Simulate an uninterruptible native read that ignores the stop event.
    while True:
        time.sleep(.1)


def test_blocked_camera_timeout_and_shutdown_are_bounded():
    capture=LatestCamera(0,worker=blocked_driver)
    start=time.perf_counter()
    try:
        with pytest.raises(ValueError,match="超时"):
            capture.read(timeout=.3)
    finally:
        capture.close()
    assert time.perf_counter()-start < 4
    assert not capture.process.is_alive()
