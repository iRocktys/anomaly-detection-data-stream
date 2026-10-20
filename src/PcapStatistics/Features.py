"""33 atributos solicitados e diversidade global; cálculos limitados à janela."""

from collections import Counter
from math import log2, sqrt

REQUESTED_FEATURES = [
    'ACK_Flag_Count', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size',
    'Bwd_Header_Length', 'Bwd_Packets_s', 'CWE_Flag_Count', 'Flow_Bytes_s',
    'Flow_Duration', 'Flow_IAT_Max', 'Flow_IAT_Mean', 'Flow_Packets_s',
    'Fwd_Header_Length', 'Fwd_IAT_Max', 'Fwd_IAT_Mean', 'Fwd_IAT_Std',
    'Fwd_IAT_Total', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Mean',
    'Fwd_Packet_Length_Min', 'Fwd_Packets_s', 'Init_Win_bytes_backward',
    'Init_Win_bytes_forward', 'Max_Packet_Length', 'Min_Packet_Length',
    'Packet_Length_Mean', 'Subflow_Bwd_Packets', 'Subflow_Fwd_Bytes',
    'Subflow_Fwd_Packets', 'Total_Backward_Packets', 'Total_Fwd_Packets',
    'Total_Length_of_Fwd_Packets', 'act_data_pkt_fwd', 'min_seg_size_forward',
]
DIVERSITY_FEATURES = ['UniqueSourceIPs', 'UniqueDestinationIPs', 'UniqueSourcePorts',
                      'UniqueDestinationPorts', 'UniqueProtocols', 'UniqueFlows',
                      'SourceIPEntropy', 'DestinationIPEntropy', 'SourcePortEntropy',
                      'DestinationPortEntropy', 'ProtocolEntropy']
FEATURE_NAMES = REQUESTED_FEATURES + DIVERSITY_FEATURES
METADATA_NAMES = ['Timestamp']


class RunningMoments:
    """Welford: média, extremos e desvio amostral sem armazenar observações."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = self.total = 0.0
        self.minimum = self.maximum = None

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def average(self):
        return self.mean if self.count else None

    def std(self):
        return sqrt(max(0.0, self.m2 / (self.count - 1))) if self.count > 1 else (0.0 if self.count else None)


def entropy(counts):
    total = sum(counts.values())
    return -sum((count / total) * log2(count / total) for count in counts.values()) if total else None


def mean_available(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


class FlowSlice:
    def __init__(self, state, timestamp):
        self.state = state
        self.first = self.last = timestamp
        self.last_forward = None
        self.subflows = 1
        self.forward_packets = self.backward_packets = self.forward_bytes = 0


class WindowAccumulator:
    def __init__(self, packet_length_mode='transport_payload', subflow_gap_ns=1_000_000_000):
        self.packet_length_mode = packet_length_mode
        self.subflow_gap_ns = subflow_gap_ns
        self.packets = 0
        self.ack = self.cwr = self.forward = self.backward = self.forward_headers = self.backward_headers = 0
        self.active_forward = 0
        self.bytes = self.forward_bytes = 0
        self.sizes, self.forward_sizes = RunningMoments(), RunningMoments()
        self.flow_iats, self.forward_iats = RunningMoments(), RunningMoments()
        self.minimum_forward_header = None
        self.source_ips, self.destination_ips = Counter(), Counter()
        self.source_ports, self.destination_ports, self.protocols = Counter(), Counter(), Counter()
        self.flows = {}

    def update(self, packet, assignment=None):
        if not packet['IsIPv4']:
            return
        self.packets += 1
        self.source_ips[packet['SourceIP']] += 1
        self.destination_ips[packet['DestinationIP']] += 1
        self.protocols[packet['Protocol']] += 1
        for key, counter in [('SourcePort', self.source_ports), ('DestinationPort', self.destination_ports)]:
            if packet[key] is not None:
                counter[packet[key]] += 1
        length = packet['IPLength'] if self.packet_length_mode == 'ip' else packet['PayloadLength']
        if length is not None:
            self.sizes.update(length)
            self.bytes += length
        flags = packet['TCPFlags'] or 0
        self.ack += bool(flags & 16)
        self.cwr += bool(flags & 128)  # Nome CWE legado; corresponde ao bit CWR.
        if assignment is None:
            return
        state, forward = assignment
        timestamp = packet['TimestampNs']
        flow = self.flows.get(state.identifier)
        if flow is None:
            flow = self.flows[state.identifier] = FlowSlice(state, timestamp)
        else:
            gap = timestamp - flow.last
            self.flow_iats.update(gap / 1000)  # microssegundos
            if gap > self.subflow_gap_ns:
                flow.subflows += 1
        flow.last = timestamp
        if forward:
            self.forward += 1
            flow.forward_packets += 1
            if length is not None:
                self.forward_bytes += length
                flow.forward_bytes += length
                self.forward_sizes.update(length)
            self.forward_headers += packet['TransportHeaderLength']
            if packet['Protocol'] == 6:
                self.active_forward += int(packet['PayloadLength'] > 0)
            if packet['Protocol'] in (6, 17):
                size = packet['TransportHeaderLength']
                self.minimum_forward_header = size if self.minimum_forward_header is None else min(self.minimum_forward_header, size)
            if flow.last_forward is not None:
                self.forward_iats.update((timestamp - flow.last_forward) / 1000)
            flow.last_forward = timestamp
        else:
            self.backward += 1
            flow.backward_packets += 1
            self.backward_headers += packet['TransportHeaderLength']

    def finish(self, seconds):
        slices = list(self.flows.values())
        result = {
            'ACK_Flag_Count': self.ack, 'CWE_Flag_Count': self.cwr,
            'Average_Packet_Size': self.sizes.average(), 'Packet_Length_Mean': self.sizes.average(),
            'Min_Packet_Length': self.sizes.minimum, 'Max_Packet_Length': self.sizes.maximum,
            'Avg_Fwd_Segment_Size': self.forward_sizes.average(),
            'Fwd_Packet_Length_Mean': self.forward_sizes.average(),
            'Fwd_Packet_Length_Min': self.forward_sizes.minimum,
            'Fwd_Packet_Length_Max': self.forward_sizes.maximum,
            'Bwd_Header_Length': self.backward_headers, 'Fwd_Header_Length': self.forward_headers,
            'Flow_Bytes_s': self.bytes / seconds, 'Flow_Packets_s': self.packets / seconds,
            'Fwd_Packets_s': self.forward / seconds, 'Bwd_Packets_s': self.backward / seconds,
            'Flow_Duration': mean_available([(flow.last - flow.first) / 1000 for flow in slices]),
            'Flow_IAT_Max': self.flow_iats.maximum, 'Flow_IAT_Mean': self.flow_iats.average(),
            'Fwd_IAT_Max': self.forward_iats.maximum, 'Fwd_IAT_Mean': self.forward_iats.average(),
            'Fwd_IAT_Std': self.forward_iats.std(), 'Fwd_IAT_Total': self.forward_iats.total,
            'Init_Win_bytes_backward': mean_available([f.state.initial_backward for f in slices if f.backward_packets]),
            'Init_Win_bytes_forward': mean_available([f.state.initial_forward for f in slices if f.forward_packets]),
            'Subflow_Bwd_Packets': sum(f.backward_packets / f.subflows for f in slices),
            'Subflow_Fwd_Bytes': sum(f.forward_bytes / f.subflows for f in slices),
            'Subflow_Fwd_Packets': sum(f.forward_packets / f.subflows for f in slices),
            'Total_Backward_Packets': self.backward, 'Total_Fwd_Packets': self.forward,
            'Total_Length_of_Fwd_Packets': self.forward_bytes,
            'act_data_pkt_fwd': self.active_forward, 'min_seg_size_forward': self.minimum_forward_header,
            'UniqueFlows': len(slices),
        }
        for prefix, counts, unique in [
            ('SourceIP', self.source_ips, 'UniqueSourceIPs'),
            ('DestinationIP', self.destination_ips, 'UniqueDestinationIPs'),
            ('SourcePort', self.source_ports, 'UniqueSourcePorts'),
            ('DestinationPort', self.destination_ports, 'UniqueDestinationPorts'),
            ('Protocol', self.protocols, 'UniqueProtocols'),
        ]:
            result[unique] = len(counts)
            result[prefix + 'Entropy'] = entropy(counts)
        return {name: result[name] for name in FEATURE_NAMES}


def feature_dictionary(packet_length_mode='transport_payload', flow_idle_timeout_seconds=120.0,
                       subflow_gap_seconds=1.0, window_seconds=1.0):
    """Único relatório persistido: definição, fórmula, unidade e adaptações."""
    length = ('L = bytes capturados após o cabeçalho TCP/UDP; ICMP desconta 8 bytes; outros protocolos usam payload IP'
              if packet_length_mode == 'transport_payload' else 'L = min(comprimento IPv4 declarado, bytes IPv4 capturados)')
    base = (f'{length}. Sem remontagem de fragmentos. Janela global de {window_seconds:g}s [início,fim), alinhada ao epoch UTC. '
            'WindowSeconds nas fórmulas é a duração configurada; PacketCount é o contador interno de IPv4. Ambos não são colunas exportadas. '
            f'Forward = orientação do primeiro pacote observado da sessão bidirecional; idle timeout = {flow_idle_timeout_seconds:g}s; '
            'RST ou FIN nas duas direções marca fechamento; orientação preservada para ACKs finais até timeout/novo SYN; SYN novo com sequência diferente ou sessão fechada reinicia. '
            'Direção persistida entre janelas/arquivos. Comprimento indisponível não vira zero. '
            'Estatística sem observações = campo vazio; contadores/somas = zero. '
            'Adaptação temporal explícita, sem garantia de igualdade com CICFlowMeter.')
    specifications = {
        'ACK_Flag_Count': ('sum(1[TCP.flags & 0x10])', 'packets', 'Pacotes com ACK, em ambas as direções.'),
        'CWE_Flag_Count': ('sum(1[TCP.flags & 0x80])', 'packets', 'Nome legado CWE corresponde a CWR, não ECE.'),
        'Average_Packet_Size': ('sum(L) / quantidade de L disponíveis', 'bytes', 'Igual a Packet_Length_Mean; não reproduz duplicações legadas de extratores.'),
        'Packet_Length_Mean': ('sum(L) / quantidade de L disponíveis', 'bytes', ''),
        'Avg_Fwd_Segment_Size': ('sum(L_forward) / quantidade de L_forward disponíveis', 'bytes', 'Alias de Fwd_Packet_Length_Mean.'),
        'Flow_Bytes_s': ('sum(L disponíveis na janela) / WindowSeconds', 'bytes/s', 'Denominador fixo da janela; não duração final dos fluxos.'),
        'Flow_Packets_s': ('PacketCount / WindowSeconds', 'packets/s', 'Todos os pacotes IPv4, inclusive sem direção recuperável.'),
        'Flow_Duration': ('mean_f((último_timestamp_f - primeiro_timestamp_f) / 1000)', 'microseconds', 'f = trecho da sessão dentro desta janela; um pacote implica duração zero. Não é idade nem duração final da conexão.'),
        'Flow_IAT_Max': ('max de (t_i - t_anterior_mesma_sessão) / 1000', 'microseconds', 'Somente pares da mesma sessão e mesma janela; exclui intervalos entre janelas e conexões distintas.'),
        'Flow_IAT_Mean': ('sum(IATs intrassessão na janela) / número desses intervalos', 'microseconds', 'Média agrupada por intervalo; sem pares entre conexões ou janelas.'),
        'Fwd_IAT_Max': ('max(IAT_forward)', 'microseconds', 'IAT_forward usa pares forward da mesma sessão e janela.'),
        'Fwd_IAT_Mean': ('sum(IAT_forward) / count(IAT_forward)', 'microseconds', ''),
        'Fwd_IAT_Std': ('sqrt(sum((IAT_forward - média)^2)/(n-1)) via Welford', 'microseconds', 'Desvio amostral: zero para n=1; vazio para n=0.'),
        'Fwd_IAT_Total': ('sum(IAT_forward)', 'microseconds', ''),
        'Fwd_Packet_Length_Max': ('max(L_forward)', 'bytes', ''),
        'Fwd_Packet_Length_Mean': ('sum(L_forward) / count(L_forward)', 'bytes', ''),
        'Fwd_Packet_Length_Min': ('min(L_forward)', 'bytes', ''),
        'Fwd_Header_Length': ('sum(comprimento cabeçalho transporte de pacotes forward)', 'bytes', 'TCP=dataofs*4; UDP=8; ICMP=8 se disponível; outros=0. Não soma cabeçalho IP.'),
        'Bwd_Header_Length': ('sum(comprimento cabeçalho transporte de pacotes backward)', 'bytes', 'Mesma convenção de Fwd_Header_Length.'),
        'Fwd_Packets_s': ('Total_Fwd_Packets / WindowSeconds', 'packets/s', ''),
        'Bwd_Packets_s': ('Total_Backward_Packets / WindowSeconds', 'packets/s', ''),
        'Init_Win_bytes_forward': ('mean_f(primeiro TCP.window observado na direção forward da sessão)', 'bytes', 'Uma contribuição por sessão com pacotes forward na janela; preserva valor entre janelas. Valor bruto sem window scaling; UDP/ICMP não contribuem.'),
        'Init_Win_bytes_backward': ('mean_f(primeiro TCP.window observado na direção backward da sessão)', 'bytes', 'Uma contribuição por sessão com pacotes backward na janela; sem window scaling.'),
        'Max_Packet_Length': ('max(L)', 'bytes', ''),
        'Min_Packet_Length': ('min(L)', 'bytes', ''),
        'Subflow_Bwd_Packets': ('sum_f(n_backward_f / k_f)', 'packets', f'k_f=1 + quantidade de gaps intrassessão > {subflow_gap_seconds:g}s dentro da janela. Não carrega contagem de subfluxos de janelas anteriores.'),
        'Subflow_Fwd_Bytes': ('sum_f(bytes_forward_f / k_f)', 'bytes', 'Mesma regra k_f; preserva resultados fracionários.'),
        'Subflow_Fwd_Packets': ('sum_f(n_forward_f / k_f)', 'packets', 'Com janela <= limiar de subfluxo, k_f=1: pode coincidir com Total_Fwd_Packets.'),
        'Total_Backward_Packets': ('sum(1[pacote atribuível à direção backward])', 'packets', ''),
        'Total_Fwd_Packets': ('sum(1[pacote atribuível à direção forward])', 'packets', ''),
        'Total_Length_of_Fwd_Packets': ('sum(L_forward disponíveis)', 'bytes', ''),
        'act_data_pkt_fwd': ('sum(1[TCP forward com payload capturado > 0])', 'packets', 'UDP e cabeçalhos TCP sem dados não contribuem.'),
        'min_seg_size_forward': ('min(cabeçalho transporte TCP/UDP forward)', 'bytes', 'TCP=dataofs*4 incluindo opções; UDP=8; não mínimo de payload. Protocolos sem TCP/UDP não contribuem.'),
        'UniqueFlows': ('número de sessões bidirecionais com pacotes atribuíveis na janela', 'count', 'TCP/UDP: 5-tupla bidirecional; demais protocolos: par de IPs e protocolo, sem portas. Identificador de sessão respeita timeout/fechamento.'),
    }
    for suffix, source in [('SourceIPs', 'IP origem'), ('DestinationIPs', 'IP destino'),
                           ('SourcePorts', 'porta origem'), ('DestinationPorts', 'porta destino'),
                           ('Protocols', 'protocolo IP')]:
        specifications['Unique' + suffix] = (f'cardinalidade dos valores distintos de {source}', 'count', 'Ocorrências por pacote; portas somente onde cabeçalho disponível.')
    for prefix, source in [('SourceIP', 'IP origem'), ('DestinationIP', 'IP destino'),
                           ('SourcePort', 'porta origem'), ('DestinationPort', 'porta destino'), ('Protocol', 'protocolo')]:
        specifications[prefix + 'Entropy'] = (f'-sum_v(p_v*log2(p_v)); p_v=count({source}=v)/N_válidos', 'bits', 'Distribuição por pacote; não por fluxo. Zero com um valor distinto; vazio sem observações.')
    rows = []
    for name in FEATURE_NAMES:
        formula, unit, notes = specifications[name]
        rows.append(dict(Feature=name, Kind='feature', Formula=formula, Unit=unit, Notes=notes, Conventions=base))
    rows.append(dict(Feature='Timestamp', Kind='metadata',
                     Formula='Término nominal exclusivo da janela, em UTC; início = Timestamp - duração configurada.',
                     Unit='UTC', Notes='Não utilizar como atributo do modelo. Primeira e última linha podem representar janelas incompletas.',
                     Conventions=base))
    return rows
