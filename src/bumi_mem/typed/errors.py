"""Errors intentionally avoid returning private record contents."""
class TypedMemoryError(RuntimeError):
    pass

class AccessDenied(TypedMemoryError):
    pass

class ConflictError(TypedMemoryError):
    pass

class EvidenceError(TypedMemoryError):
    pass

class SchemaMismatch(TypedMemoryError):
    pass

class StaleRead(TypedMemoryError):
    pass
