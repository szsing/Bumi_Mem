"""测试全局 fixtures：确保每个测试后关闭新建的引擎实例。"""
import pytest


@pytest.fixture(autouse=True)
def close_test_engines():
    """自动 fixture：测试结束后关闭该测试中新建的 BumiMemory 引擎，防止资源泄漏。"""
    from test_bumi_memory import ENGINES
    start = len(ENGINES)
    yield
    for engine in ENGINES[start:]:
        engine.close()
    del ENGINES[start:]