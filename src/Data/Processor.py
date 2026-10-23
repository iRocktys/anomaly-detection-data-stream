import numpy as np
import pandas as pd
from capymoa.stream import NumpyStream
from src.Scenario.FlowCSVReader import FlowCSVReader

from src.Data.FeatureAudit import buildFeatureRemovalReport, printFeatureRemovalReport


class DataStreamProcessor:
    metadataColumns = {
        "Timestamp", "OriginalTimestamp", "FlowStart", "FlowEnd", "SourceFile", "SourceRow",
        "SourceDataset", "SourceWindowRow", "InstanceIndex", "InstanceId", "BlockIndex",
        "ScenarioPart", "AttackBlock", "TemporalStratum", "SamplingSeed", "WindowStart",
        "WindowEnd", "WindowSeconds", "TargetAttack", "Attack", "Flow_ID", "Source_IP",
        "Destination_IP", "SimillarHTTP", "AttackCount", "BenignCount", "OtherAttackCount",
        "AttackRatio", "RepeatOfBlock", "RepeatAttackBlocks",
    }

    # Inicializa o processador com seleção e remoção opcionais de features.
    def __init__(self, logging=True, selected_features=None, removed_features=None):
        self.logging = bool(logging)
        self.selected_features = None if selected_features is None else list(dict.fromkeys(FlowCSVReader.normalizeColumnName(f) for f in selected_features))
        self.removed_features = None if removed_features is None else list(dict.fromkeys(FlowCSVReader.normalizeColumnName(f) for f in removed_features))
        self.featureRemovalRecords = []

    # Exibe mensagens de processamento somente quando o logging estiver habilitado.
    def _log(self, message):
        if self.logging:
            print(message)

    # Registra os nomes das características efetivamente excluídas de uma etapa.
    def _record_feature_removals(self, stage, features, reason):
        self.featureRemovalRecords.extend({"Stage": stage, "Feature": feature, "Reason": reason} for feature in features)

    # Retorna o relatório completo de exclusões aplicado à construção da stream.
    def getFeatureRemovalReport(self):
        return buildFeatureRemovalReport(self.featureRemovalRecords)

    # Exibe o relatório de exclusões uma única vez por preparação do dataset.
    def reportFeatureRemovals(self, title="Features removidas da stream"):
        printFeatureRemovalReport(self.getFeatureRemovalReport(), title)

    # Valida o DataFrame de entrada e a presença da coluna de rótulo.
    def _validate_dataframe(self, df, target_label_col):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The dataset must be a pandas DataFrame.")
        if target_label_col not in df.columns:
            raise ValueError(f"Target column '{target_label_col}' was not found in the dataset.")

    # Remove linhas sem rótulo válido antes de construir a stream binária.
    def _remove_invalid_label_rows(self, df, target_label_col):
        normalized_labels = df[target_label_col].astype("string").str.strip()
        invalid_mask = normalized_labels.isna() | normalized_labels.fillna("").eq("")
        cleaned_df = df.loc[~invalid_mask].copy().reset_index(drop=True)
        return cleaned_df, int(invalid_mask.sum())

    # Mantém somente as features selecionadas e a coluna de rótulo quando houver seleção explícita.
    def _select_features(self, df, target_label_col):
        if self.selected_features is None:
            return df.copy()
        selected_features = [str(feature).strip() for feature in self.selected_features]
        missing_features = [feature for feature in selected_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Selected features were not found in the dataset: {missing_features}")
        forbidden = set(selected_features) & self.metadataColumns
        if forbidden:
            raise ValueError(f"Metadados não podem ser selecionados como features: {sorted(forbidden)}")
        columns_to_keep = list(selected_features)
        if target_label_col not in columns_to_keep:
            columns_to_keep.append(target_label_col)
        excluded = [column for column in df if column not in columns_to_keep]
        self._record_feature_removals("Seleção explícita", excluded, "Fora da lista selected_features")
        return df[columns_to_keep].copy()

    # Remove somente features existentes sem permitir a exclusão da coluna de rótulo.
    def _remove_features(self, df, target_label_col):
        if self.removed_features is None:
            return df
        removed_features = [str(feature).strip() for feature in self.removed_features]
        removed_features = [feature for feature in removed_features if feature != target_label_col]
        existing_features = [feature for feature in removed_features if feature in df.columns]
        self._record_feature_removals("Remoção explícita", existing_features, "Informada em removed_features")
        return df.drop(columns=existing_features)

    # Converte as features para valores numéricos preservando ausências para a imputação online.
    def _prepare_features(self, df, target_label_col):
        metadata = [column for column in df if column in self.metadataColumns
                    or column.startswith("Unnamed") or column.lower() == "index"]
        self._record_feature_removals("Metadados", metadata, "Metadado excluído dos preditores")
        self._record_feature_removals("Alvo", [target_label_col], "Alvo supervisionado; não é um preditor")
        features = df.drop(columns=metadata + [target_label_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)
        if features.shape[1] == 0:
            raise ValueError("No feature remained after dataset preprocessing.")
        return features.astype(np.float64)

    # Normaliza os rótulos originais preservando o nome real do ataque apenas como metadado.
    def _prepare_original_labels(self, labels):
        original_labels = labels.astype("string").str.strip()
        invalid_mask = original_labels.isna() | original_labels.fillna("").eq("")
        if invalid_mask.any():
            invalid_indices = original_labels.index[invalid_mask].tolist()
            raise ValueError(f"Missing or empty labels were found at rows {invalid_indices[:10]}.")
        return original_labels.astype(str)

    # Converte BENIGN ou NORMAL para zero e qualquer ataque para um no alvo experimental.
    def _encode_binary_labels(self, original_labels):
        normal_labels = {"BENIGN", "NORMAL", "0"}
        encoded_labels = np.where(
            original_labels.str.upper().isin(normal_labels),
            0,
            1,
        ).astype(np.int32)
        return encoded_labels, ["BENIGN", "ATTACK"]

    # Constrói uma NumpyStream binária mantendo nomes de features e rótulos originais alinhados.
    def create_stream(self, df, target_label_col="Label", binary_label=True):
        if not bool(binary_label):
            raise ValueError("Only binary stream encoding is supported; binary_label must remain True.")
        self.featureRemovalRecords = []
        processed_df = df.copy()
        processed_df.columns = [FlowCSVReader.normalizeColumnName(c) for c in processed_df.columns]
        if not processed_df.columns.is_unique:
            raise ValueError("Colunas duplicadas após normalização dos nomes.")
        target_label_col = FlowCSVReader.normalizeColumnName(target_label_col)
        self._validate_dataframe(processed_df, target_label_col)
        processed_df, removed_label_rows = self._remove_invalid_label_rows(processed_df, target_label_col)
        self._log(f"Linhas removidas por rótulo ausente ou vazio: {removed_label_rows}.")
        if processed_df.empty:
            raise ValueError("All rows were removed because no valid label remained.")
        processed_df = self._select_features(processed_df, target_label_col)
        processed_df = self._remove_features(processed_df, target_label_col)
        original_labels = self._prepare_original_labels(processed_df[target_label_col])
        features = self._prepare_features(processed_df, target_label_col)
        if self.logging:
            self.reportFeatureRemovals()
        encoded_labels, target_names = self._encode_binary_labels(original_labels)
        feature_names = features.columns.tolist()
        feature_values = features.to_numpy(dtype=np.float64)
        label_names = original_labels.tolist()
        if len(feature_values) != len(encoded_labels) or len(encoded_labels) != len(label_names):
            raise RuntimeError("Feature values and binary labels have different lengths.")
        stream = NumpyStream(
            feature_values,
            encoded_labels,
            target_name=target_label_col,
            feature_names=feature_names,
            target_type="categorical",
        )
        self._log(f"Stream criada com {feature_values.shape[0]} instâncias e {feature_values.shape[1]} features.")
        self._log("Treinamento configurado exclusivamente como binário: BENIGN=0 e ATTACK=1.")
        self._log(f"Valores ausentes preservados para imputação online: {int(np.isnan(feature_values).sum())}.")
        return stream, target_names, feature_names, label_names
