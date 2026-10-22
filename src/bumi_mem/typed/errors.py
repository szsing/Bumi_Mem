"""类型化记忆错误类型：故意不返回私有记录内容，防止信息泄露。"""


class TypedMemoryError(RuntimeError):
    """类型化记忆基础错误。"""
    pass

class AccessDenied(TypedMemoryError):
    """访问拒绝错误：策略引擎判定请求者无权访问。"""
    pass

class ConflictError(TypedMemoryError):
    """冲突错误：幂等键冲突或版本冲突。"""
    pass

class EvidenceError(TypedMemoryError):
    """证据错误：源溯源不满足记忆写入的前置条件。"""
    pass

class SchemaMismatch(TypedMemoryError):
    """模式不匹配错误：数据库模式版本与当前代码不兼容。"""
    pass

class StaleRead(TypedMemoryError):
    """过期读取错误：读取时代数已过期，需重新获取。"""
    pass