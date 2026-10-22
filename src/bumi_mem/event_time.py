"""验证可信事件时间戳；缺失时间戳表示服务器接收时间。"""
from datetime import datetime


def message_time(value: str | None = None) -> str:
    if value is None:
        return datetime.now().astimezone().isoformat()
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError('event time must be an offset-aware ISO 8601 string')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('event time must be an offset-aware ISO 8601 string') from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('event time must include a timezone offset')
    return parsed.isoformat()
