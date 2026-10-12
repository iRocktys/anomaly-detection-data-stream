import numpy as np


class LabelTools:
    @staticmethod
    def find_normal_class_idx(target_names):
        for idx, name in enumerate(target_names):
            if str(name).strip().upper() in ["BENIGN", "NORMAL", "0"]:
                return idx

        return 0

    @staticmethod
    def to_binary_label(label, normal_class_idx=0):
        return 0 if int(label) == int(normal_class_idx) else 1

    @staticmethod
    def to_binary_labels(labels, normal_class_idx=0):
        labels = np.asarray(labels)

        return np.array(
            [
                LabelTools.to_binary_label(label, normal_class_idx)
                for label in labels
            ],
            dtype=int,
        )

    @staticmethod
    def extract_attack_regions(
        y_true_multi,
        normal_class_idx=0,
        max_gap_between_attacks=1000,
    ):
        y_true_array = np.asarray(y_true_multi)
        attack_indices = np.where(y_true_array != normal_class_idx)[0]

        attack_regions = []

        if len(attack_indices) == 0:
            return attack_regions

        start_idx = attack_indices[0]
        last_idx = attack_indices[0]

        for idx in attack_indices[1:]:
            if idx - last_idx > max_gap_between_attacks:
                block_label = LabelTools._dominant_attack_label(
                    y_true_array=y_true_array,
                    start_idx=start_idx,
                    end_idx=last_idx,
                    normal_class_idx=normal_class_idx,
                )

                attack_regions.append((start_idx, last_idx, block_label))
                start_idx = idx

            last_idx = idx

        block_label = LabelTools._dominant_attack_label(
            y_true_array=y_true_array,
            start_idx=start_idx,
            end_idx=last_idx,
            normal_class_idx=normal_class_idx,
        )

        attack_regions.append((start_idx, last_idx, block_label))

        return attack_regions

    @staticmethod
    def _dominant_attack_label(
        y_true_array,
        start_idx,
        end_idx,
        normal_class_idx,
    ):
        block_labels = y_true_array[start_idx:end_idx + 1]
        attack_labels = block_labels[block_labels != normal_class_idx]

        if len(attack_labels) == 0:
            return 1

        return int(np.bincount(attack_labels.astype(int)).argmax())