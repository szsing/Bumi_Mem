"""Small user-ID boundary and volatile frontend identity binding; no voice recognition."""
from dataclasses import dataclass
import threading
from uuid import uuid4
from .schemas import IdentityRequiredError


GUEST_USER_ID = "__guest__"


def is_guest(user_id: str) -> bool:
    """One reserved unknown-speaker category, never a persistent person."""
    return user_id == GUEST_USER_ID


def require_person(user_id: str) -> str:
    require_id(user_id)
    if is_guest(user_id):
        raise IdentityRequiredError('guest identity has no personal long-term memory')
    return user_id


def require_id(value, label='user_id'):
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128 or not value.isprintable():
        raise IdentityRequiredError(f'{label} must be an explicit nonempty stable ID')
    return value


def require_pair(user_id, agent_id):
    return require_id(user_id), require_id(agent_id, 'agent_id')


@dataclass(frozen=True, slots=True)
class IdentityTicket:
    user_id: str
    generation: int
    connection_id: str


class UserIdentity:
    """Caller supplies an identity already confirmed by a trusted frontend.

    A new instance/reconnection starts unbound. Do not persist the active speaker.
    Async consumers must check the ticket at final display, not merely at enqueue.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._connection = uuid4().hex
        self._generation = 0
        self._user = None

    def bind(self, user_id):
        user_id = require_id(user_id)
        with self._lock:
            self._generation += 1
            self._user = user_id
            return self.snapshot()

    def clear(self):
        with self._lock:
            self._generation += 1
            self._user = None

    def snapshot(self):
        with self._lock:
            if self._user is None:
                raise IdentityRequiredError('no user has been identified')
            return IdentityTicket(self._user, self._generation, self._connection)

    def accepts(self, ticket):
        with self._lock:
            return (self._user is not None and ticket.user_id == self._user
                    and ticket.generation == self._generation and ticket.connection_id == self._connection)

    def deliver(self, ticket, callback, value):
        with self._lock:
            if not self.accepts(ticket):
                return False
            callback(value)
            return True
