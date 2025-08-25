from pathlib import Path

import pytest


@pytest.fixture
def ROOT_DIR():
    return Path(__file__).parent.parent
