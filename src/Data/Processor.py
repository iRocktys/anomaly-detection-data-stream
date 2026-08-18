import numpy as np
import pandas as pd
from capymoa.stream import NumpyStream


class DataStreamProcessor:
    def __init__(
        self,
        logging=True,
        selected_features=None,
        removed_features=None,
    ):
        self.logging = logging
        self.selected_features = selected_features
        self.removed_features = removed_features

    def _log(self, message):
        if self.logging:
            print(message)

    def _validate_dataframe(self, df, target_label_col):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "O conjunto de dados deve ser um DataFrame do pandas."
            )

        if target_label_col not in df.columns:
            raise ValueError(
                f"A coluna de rótulo '{target_label_col}' "
                "não foi encontrada no dataset."
            )

    def _remove_invalid_label_rows(self, df, target_label_col):
        normalized_labels = (
            df[target_label_col]
            .astype("string")
            .str.strip()
        )

        invalid_mask = (
            normalized_labels.isna()
            | normalized_labels.fillna("").eq("")
        )

        removed_rows = int(invalid_mask.sum())

        cleaned_df = (
            df.loc[~invalid_mask]
            .copy()
            .reset_index(drop=True)
        )

        return cleaned_df, removed_rows

    def _select_features(self, df, target_label_col):
        if self.selected_features is None:
            return df.copy()

        selected_features = [
            str(feature).strip()
            for feature in self.selected_features
        ]

        missing_features = [
            feature
            for feature in selected_features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                "As seguintes features selecionadas não foram encontradas: "
                f"{missing_features}"
            )

        columns_to_keep = selected_features.copy()

        if target_label_col not in columns_to_keep:
            columns_to_keep.append(target_label_col)

        return df[columns_to_keep].copy()

    def _remove_features(self, df, target_label_col):
        if self.removed_features is None:
            return df

        removed_features = [
            str(feature).strip()
            for feature in self.removed_features
        ]

        removed_features = [
            feature
            for feature in removed_features
            if feature != target_label_col
        ]

        existing_features = [
            feature
            for feature in removed_features
            if feature in df.columns
        ]

        return df.drop(columns=existing_features)

    def _prepare_features(self, df, target_label_col):
        features = df.drop(columns=[target_label_col])

        features = features.apply(
            pd.to_numeric,
            errors="coerce",
        )

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if features.shape[1] == 0:
            raise ValueError(
                "Nenhuma feature permaneceu disponível "
                "após o processamento."
            )

        return features.astype(np.float64)

    def _prepare_original_labels(self, labels):
        original_labels = (
            labels
            .astype("string")
            .str.strip()
        )

        invalid_mask = (
            original_labels.isna()
            | original_labels.fillna("").eq("")
        )

        if invalid_mask.any():
            invalid_indices = (
                original_labels
                .index[invalid_mask]
                .tolist()
            )

            preview = invalid_indices[:10]
            suffix = (
                "..."
                if len(invalid_indices) > len(preview)
                else ""
            )

            raise ValueError(
                "Foram encontrados rótulos ausentes ou vazios "
                f"nas linhas {preview}{suffix}. "
                "Rótulos não podem ser imputados nem tratados "
                "como ataques."
            )

        return original_labels.astype(str)

    def _encode_labels(
        self,
        original_labels,
        binary_label=True,
    ):
        if binary_label:
            normal_labels = [
                "BENIGN",
                "NORMAL",
            ]

            encoded_labels = np.where(
                original_labels
                .str.upper()
                .isin(normal_labels),
                0,
                1,
            ).astype(np.int32)

            target_names = [
                "BENIGN",
                "ATTACK",
            ]

            return encoded_labels, target_names

        unique_labels = (
            original_labels
            .drop_duplicates()
            .tolist()
        )

        normal_label = next(
            (
                label
                for label in unique_labels
                if label.upper() in ["BENIGN", "NORMAL"]
            ),
            None,
        )

        if normal_label is not None:
            unique_labels.remove(normal_label)
            unique_labels.insert(0, normal_label)

        label_mapping = {
            label: index
            for index, label in enumerate(unique_labels)
        }

        encoded_labels = (
            original_labels
            .map(label_mapping)
            .to_numpy(dtype=np.int32)
        )

        return encoded_labels, unique_labels

    def create_stream(
        self,
        df,
        target_label_col="Label",
        binary_label=True,
    ):
        processed_df = df.copy()

        processed_df.columns = (
            processed_df.columns
            .astype(str)
            .str.strip()
        )

        target_label_col = str(target_label_col).strip()

        self._validate_dataframe(
            processed_df,
            target_label_col,
        )

        processed_df, removed_label_rows = (
            self._remove_invalid_label_rows(
                processed_df,
                target_label_col,
            )
        )

        self._log(
            "Linhas removidas por rótulo ausente ou vazio: "
            f"{removed_label_rows}."
        )

        if processed_df.empty:
            raise ValueError(
                "Todas as linhas foram removidas porque "
                "não possuem rótulo válido."
            )

        processed_df = self._select_features(
            processed_df,
            target_label_col,
        )

        processed_df = self._remove_features(
            processed_df,
            target_label_col,
        )

        original_labels = self._prepare_original_labels(
            processed_df[target_label_col]
        )

        features = self._prepare_features(
            processed_df,
            target_label_col,
        )

        encoded_labels, target_names = self._encode_labels(
            original_labels,
            binary_label,
        )

        feature_names = features.columns.tolist()
        feature_values = features.to_numpy(dtype=np.float64)
        label_names = original_labels.tolist()

        if (
            len(feature_values) != len(encoded_labels)
            or len(encoded_labels) != len(label_names)
        ):
            raise RuntimeError(
                "Features, rótulos binários e rótulos originais "
                "possuem tamanhos diferentes."
            )

        stream = NumpyStream(
            feature_values,
            encoded_labels,
            target_name=target_label_col,
            feature_names=feature_names,
            target_type="categorical",
        )

        self._log(
            f"Stream criada com {feature_values.shape[0]} "
            f"instâncias e {feature_values.shape[1]} features."
        )

        self._log(
            "Treinamento: "
            f"{'binário' if binary_label else 'multiclasse'}."
        )

        self._log(
            "Rótulos originais preservados: "
            f"{len(set(label_names))} classes."
        )

        self._log(
            "Valores ausentes preservados para imputação online: "
            f"{int(np.isnan(feature_values).sum())}."
        )

        return (
            stream,
            target_names,
            feature_names,
            label_names,
        )