import numpy as np
import pandas as pd


class DataStreamProcessor:
    def __init__(self, logging=True, selected_features=None, removed_features=None, normal_labels=None):
        # Inicializa o processador responsável por converter um DataFrame em stream binária.
        self.logging = bool(logging)
        self.selected_features = selected_features
        self.removed_features = removed_features
        self.normal_labels = tuple(str(value).strip().upper() for value in (normal_labels or ("BENIGN", "NORMAL", "0")))

    def _log(self, message):
        # Exibe mensagens de preparação somente quando o logging está habilitado.
        if self.logging:
            print(message)

    def _validate_dataframe(self, dataframe, target_column):
        # Valida o tipo do dataset e a presença da coluna binária de rótulo.
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("The dataset must be a pandas DataFrame.")
        if target_column not in dataframe.columns:
            raise ValueError(f"The label column '{target_column}' was not found in the dataset.")

    def _remove_invalid_label_rows(self, dataframe, target_column):
        # Remove linhas sem rótulo antes de qualquer transformação de features.
        normalized_labels = dataframe[target_column].astype("string").str.strip()
        invalid_mask = normalized_labels.isna() | normalized_labels.fillna("").eq("")
        cleaned_dataframe = dataframe.loc[~invalid_mask].copy().reset_index(drop=True)
        return cleaned_dataframe, int(invalid_mask.sum())

    def _select_features(self, dataframe, target_column):
        # Mantém somente as features explicitamente selecionadas e a coluna de rótulo.
        if self.selected_features is None:
            return dataframe.copy()
        selected_features = [str(feature).strip() for feature in self.selected_features]
        missing_features = [feature for feature in selected_features if feature not in dataframe.columns]
        if missing_features:
            raise ValueError(f"The following selected features were not found: {missing_features}")
        columns_to_keep = list(dict.fromkeys(selected_features + [target_column]))
        return dataframe[columns_to_keep].copy()

    def _remove_features(self, dataframe, target_column):
        # Remove features opcionais sem permitir a exclusão da coluna de rótulo.
        if self.removed_features is None:
            return dataframe
        removed_features = [str(feature).strip() for feature in self.removed_features if str(feature).strip() != target_column]
        existing_features = [feature for feature in removed_features if feature in dataframe.columns]
        return dataframe.drop(columns=existing_features)

    def _prepare_features(self, dataframe, target_column):
        # Converte todas as features para float e preserva ausências para a imputação online.
        features = dataframe.drop(columns=[target_column]).apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)
        if features.shape[1] == 0:
            raise ValueError("No feature remained available after processing.")
        return features.astype(np.float64)

    def _encode_binary_labels(self, labels):
        # Converte qualquer rótulo normal conhecido em zero e todos os demais em ataque igual a um.
        normalized_labels = labels.astype("string").str.strip()
        encoded_labels = np.where(normalized_labels.str.upper().isin(self.normal_labels), 0, 1).astype(np.int32)
        binary_names = np.where(encoded_labels == 0, "BENIGN", "ATTACK").tolist()
        return encoded_labels, binary_names

    def create_stream(self, dataframe, target_label_col="Label"):
        # Cria uma NumpyStream estritamente binária e retorna metadados necessários ao pipeline.
        from capymoa.stream import NumpyStream

        processed_dataframe = dataframe.copy()
        processed_dataframe.columns = processed_dataframe.columns.astype(str).str.strip()
        target_column = str(target_label_col).strip()
        self._validate_dataframe(processed_dataframe, target_column)
        processed_dataframe, removed_rows = self._remove_invalid_label_rows(processed_dataframe, target_column)
        self._log(f"Linhas removidas por rótulo ausente ou vazio: {removed_rows}.")
        if processed_dataframe.empty:
            raise ValueError("All rows were removed because they do not contain a valid label.")
        processed_dataframe = self._select_features(processed_dataframe, target_column)
        processed_dataframe = self._remove_features(processed_dataframe, target_column)
        features = self._prepare_features(processed_dataframe, target_column)
        encoded_labels, binary_names = self._encode_binary_labels(processed_dataframe[target_column])
        feature_names = features.columns.tolist()
        feature_values = features.to_numpy(dtype=np.float64)
        stream = NumpyStream(
            feature_values,
            encoded_labels,
            target_name=target_column,
            feature_names=feature_names,
            target_type="categorical",
        )
        self._log(f"Stream binária criada com {len(feature_values)} instâncias e {len(feature_names)} features.")
        self._log(f"Ataques: {int(encoded_labels.sum())} | Benignos: {int(len(encoded_labels) - encoded_labels.sum())}.")
        self._log(f"Valores ausentes preservados para imputação online: {int(np.isnan(feature_values).sum())}.")
        return stream, tuple(feature_names), tuple(binary_names), encoded_labels.copy()
