"""28 estatísticas globais de IPv4 por janela, sem estado de conexão."""
from collections import Counter
from math import log2, sqrt

FEATURE_NAMES = [
    'Packet_Count', 'IP_Bytes_Total',
    'Packet_Length_Min', 'Packet_Length_Max', 'Packet_Length_Mean', 'Packet_Length_Std',
    'IAT_Min', 'IAT_Max', 'IAT_Mean', 'IAT_Std',
    'TCP_Ratio', 'UDP_Ratio', 'ICMP_Ratio',
    'SYN_Ratio', 'ACK_Ratio', 'RST_Ratio', 'FIN_Ratio', 'SYN_No_ACK_Ratio',
    'Src_IP_Unique', 'Dst_IP_Unique', 'Src_Port_Unique', 'Dst_Port_Unique',
    'Src_IP_Entropy', 'Dst_IP_Entropy', 'Src_Port_Entropy', 'Dst_Port_Entropy',
    'TTL_Mean', 'TTL_Std',
]
METADATA_NAMES = ['WindowStart', 'Timestamp', 'WindowComplete']


class RunningMoments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.0
        self.minimum = self.maximum = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.n == 1 else min(self.minimum, value)
        self.maximum = value if self.n == 1 else max(self.maximum, value)

    def values(self):
        # Desvio populacional da janela; zero quando não há observações.
        return self.minimum, self.maximum, self.mean, sqrt(max(0.0, self.m2 / self.n)) if self.n else 0.0


def entropy(counter):
    total = sum(counter.values())
    return -sum((n / total) * log2(n / total) for n in counter.values()) if total else 0.0


class WindowAccumulator:
    def __init__(self):
        self.packets = self.bytes = 0
        self.sizes, self.iats, self.ttls = RunningMoments(), RunningMoments(), RunningMoments()
        self.previous = None
        self.protocols, self.flags = Counter(), Counter()
        self.diversity = {name: Counter() for name in ('Src_IP', 'Dst_IP', 'Src_Port', 'Dst_Port')}

    def update(self, packet):
        if not packet['IsIPv4']:
            return
        self.packets += 1
        self.bytes += packet['IPLength']
        self.sizes.update(packet['IPLength'])
        self.ttls.update(packet['TTL'])
        ns = packet['TimestampNs']
        if self.previous is not None:
            if ns < self.previous:
                raise ValueError('IAT negativo: pacotes fora de ordem.')
            self.iats.update((ns - self.previous) / 1000.0)
        self.previous = ns
        self.protocols[packet['Protocol']] += 1
        for name, field in [('Src_IP', 'SourceIP'), ('Dst_IP', 'DestinationIP'),
                            ('Src_Port', 'SourcePort'), ('Dst_Port', 'DestinationPort')]:
            if packet[field] is not None:
                self.diversity[name][packet[field]] += 1
        flags = packet['TCPFlags']
        if flags is not None:
            for name, mask in [('SYN', 2), ('ACK', 16), ('RST', 4), ('FIN', 1)]:
                self.flags[name] += bool(flags & mask)
            self.flags['SYN_No_ACK'] += bool(flags & 2 and not flags & 16)

    def finish(self):
        row = dict(Packet_Count=self.packets, IP_Bytes_Total=self.bytes)
        for prefix, moments in [('Packet_Length', self.sizes), ('IAT', self.iats)]:
            for suffix, value in zip(('Min', 'Max', 'Mean', 'Std'), moments.values()):
                row[f'{prefix}_{suffix}'] = value
        for name, number in [('TCP', 6), ('UDP', 17), ('ICMP', 1)]:
            row[f'{name}_Ratio'] = self.protocols[number] / self.packets if self.packets else 0.0
        for name in ('SYN', 'ACK', 'RST', 'FIN', 'SYN_No_ACK'):
            row[f'{name}_Ratio'] = self.flags[name] / self.protocols[6] if self.protocols[6] else 0.0
        for name, counter in self.diversity.items():
            row[name + '_Unique'] = len(counter)
            row[name + '_Entropy'] = entropy(counter)
        row['TTL_Mean'], row['TTL_Std'] = self.ttls.values()[2:]
        return {name: row[name] for name in FEATURE_NAMES}
