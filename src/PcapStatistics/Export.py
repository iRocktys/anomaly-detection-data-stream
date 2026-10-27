"""Aplica a regra de rótulos às contagens salvas, sem reler PCAPs."""
import csv
from itertools import zip_longest
import json
from pathlib import Path


def export_labeled_dataset(features_path, counts_path, output_path, minimum_attack_packets=1,
                           minimum_attack_ratio=0.0):
    if isinstance(minimum_attack_packets, bool) or not isinstance(minimum_attack_packets, int) or minimum_attack_packets < 1:
        raise ValueError('minimum_attack_packets deve ser inteiro >= 1.')
    if not 0 <= minimum_attack_ratio <= 1:
        raise ValueError('minimum_attack_ratio deve estar entre 0 e 1.')
    output = Path(output_path).resolve()
    if output in (Path(features_path).resolve(), Path(counts_path).resolve()):
        raise ValueError('A saída final deve ser distinta dos intermediários.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    distribution = {}
    try:
        with open(features_path, newline='', encoding='utf-8') as fh, open(counts_path, newline='', encoding='utf-8') as ch, temporary.open('w', newline='', encoding='utf-8') as oh:
            features, counts = csv.DictReader(fh), csv.DictReader(ch)
            writer = csv.DictWriter(oh, fieldnames=features.fieldnames + ['Label', 'BinaryLabel'])
            writer.writeheader()
            for feature, count in zip_longest(features, counts):
                if feature is None or count is None or feature['Timestamp'] != count['Timestamp'] or feature['WindowStart'] != count['WindowStart']:
                    raise ValueError('Intermediários possuem janelas diferentes.')
                n = int(feature['Packet_Count'])
                benign, attack, unknown = (int(count[k]) for k in ('Benign_Packet_Count', 'Attack_Packet_Count', 'Unknown_Packet_Count'))
                if n != benign + attack + unknown:
                    raise ValueError('Contagens de rótulos não conciliam com Packet_Count.')
                attacks = json.loads(count['Attack_Class_Counts'])
                if sum(attacks.values()) != attack:
                    raise ValueError('Contagens por classe inconsistentes.')
                if n == 0:
                    label, binary = 'EMPTY', ''
                elif attack >= minimum_attack_packets and attack / n >= minimum_attack_ratio:
                    label, binary = (next(iter(attacks)) if len(attacks) == 1 else 'MULTIPLE'), 1
                elif benign == n:
                    label, binary = 'BENIGN', 0
                else:
                    # Ataque abaixo do limiar continua não sendo benigno.
                    label, binary = 'UNKNOWN', ''
                feature.update(Label=label, BinaryLabel=binary)
                writer.writerow(feature)
                distribution[label] = distribution.get(label, 0) + 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(datasetPath=str(output), labelCounts=distribution)
