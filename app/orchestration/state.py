from enum import Enum


class ConversationState(Enum):
    IDLE = "idle"
    FACE_DETECTED = "face_detected"
    GREETING = "greeting"
    CONVERSING = "conversing"
    REGISTERING = "registering"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    WAITING = "waiting"
    SESSION_TIMEOUT = "session_timeout"


VALID_TRANSITIONS = {
    ConversationState.IDLE: [ConversationState.FACE_DETECTED, ConversationState.LISTENING, ConversationState.WAITING],
    ConversationState.FACE_DETECTED: [ConversationState.GREETING, ConversationState.CONVERSING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.GREETING: [ConversationState.CONVERSING, ConversationState.REGISTERING, ConversationState.LISTENING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.CONVERSING: [ConversationState.LISTENING, ConversationState.SPEAKING, ConversationState.PROCESSING, ConversationState.WAITING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.REGISTERING: [ConversationState.LISTENING, ConversationState.SPEAKING, ConversationState.PROCESSING, ConversationState.IDLE],
    ConversationState.LISTENING: [ConversationState.SPEAKING, ConversationState.PROCESSING, ConversationState.IDLE, ConversationState.REGISTERING, ConversationState.SESSION_TIMEOUT],
    ConversationState.SPEAKING: [ConversationState.LISTENING, ConversationState.CONVERSING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.PROCESSING: [ConversationState.SPEAKING, ConversationState.CONVERSING, ConversationState.LISTENING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.WAITING: [ConversationState.LISTENING, ConversationState.GREETING, ConversationState.IDLE, ConversationState.SESSION_TIMEOUT],
    ConversationState.SESSION_TIMEOUT: [ConversationState.IDLE, ConversationState.FACE_DETECTED],
}

STATE_NAMES = {s: s.value for s in ConversationState}


class ConversationStateMachine:
    def __init__(self, event_bus=None):
        self.state = ConversationState.IDLE
        self._listeners = []
        self.event_bus = event_bus

    def can_transition(self, new_state: ConversationState) -> bool:
        return new_state in VALID_TRANSITIONS.get(self.state, [])

    async def transition(self, new_state: ConversationState) -> bool:
        if not self.can_transition(new_state):
            return False
        old = self.state
        self.state = new_state
        if self.event_bus and old != new_state:
            await self.event_bus.publish("state_changed", {"old": old.value, "new": new_state.value})
        return True

    def is_idle(self) -> bool:
        return self.state == ConversationState.IDLE

    def can_greet(self) -> bool:
        return self.state in (ConversationState.IDLE, ConversationState.FACE_DETECTED)

    def can_register(self) -> bool:
        return self.state == ConversationState.IDLE or (
            self.state == ConversationState.CONVERSING and self.can_transition(ConversationState.REGISTERING)
        )
