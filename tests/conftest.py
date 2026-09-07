import pytest


@pytest.fixture(autouse=True)
def close_test_engines():
    from test_bumi_memory import ENGINES
    start = len(ENGINES)
    yield
    for engine in ENGINES[start:]:
        engine.close()
    del ENGINES[start:]
