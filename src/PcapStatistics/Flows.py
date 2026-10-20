"""Estado mínimo bidirecional; os acumuladores estatísticos pertencem à janela."""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class FlowState:
    identifier: int
    forward: tuple
    last_ns: int
    initial_forward: object = None
    initial_backward: object = None
    syn_sequence: object = None
    fin_directions: int = 0
    closed: bool = False


class FlowTracker:
    def __init__(self, idle_timeout_ns):
        self.idle_timeout_ns = idle_timeout_ns
        self.states = OrderedDict()
        self.next_identifier = 0

    def expire(self, timestamp_ns):
        cutoff = timestamp_ns - self.idle_timeout_ns
        while self.states:
            key, state = next(iter(self.states.items()))
            if state.last_ns >= cutoff:
                break
            self.states.popitem(last=False)

    def observe(self, packet):
        source = (packet['SourceIP'], packet['SourcePort'] or 0)
        destination = (packet['DestinationIP'], packet['DestinationPort'] or 0)
        key = (packet['Protocol'], *sorted((source, destination)))
        direction_key = (source, destination)
        timestamp = packet['TimestampNs']
        self.expire(timestamp)
        state = self.states.get(key)
        flags = packet['TCPFlags'] or 0
        new_syn = flags & 2 and not flags & 16
        # SYN retransmitido com a mesma sequência não abre outra sessão.
        if state is not None and new_syn and (state.closed or state.syn_sequence != packet['TCPSequence']):
            del self.states[key]
            state = None
        if state is None:
            self.next_identifier += 1
            state = FlowState(self.next_identifier, direction_key, timestamp,
                              syn_sequence=packet['TCPSequence'] if new_syn else None)
            self.states[key] = state
        forward = direction_key == state.forward
        if packet['TCPWindow'] is not None:
            attribute = 'initial_forward' if forward else 'initial_backward'
            if getattr(state, attribute) is None:
                setattr(state, attribute, packet['TCPWindow'])
        state.last_ns = timestamp
        self.states.move_to_end(key)
        if flags & 1:
            state.fin_directions |= 1 if forward else 2
        if flags & 4 or state.fin_directions == 3:
            state.closed = True
        return state, forward
