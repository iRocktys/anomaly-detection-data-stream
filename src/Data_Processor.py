import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from capymoa.stream import NumpyStream

class DataStreamProcessor:
    """
    classe responsável por processar e preparar os dados de rede para fluxos contínuos,
    aplicando limpeza, normalização e seleção de features.
    """
    
    def __init__(self, logging=True):
        self.logging = logging

    def _log(self, message):
        if self.logging:
            print(message)

    def _remove_features(self, X, y, threshold_var=None, threshold_corr=None, top_n_features=None):
        initial_count = X.shape[1]
        self._log(f"\n--- Iniciando Processo de Seleção de Features (Total: {initial_count}) ---")

        # remoção por variância 
        if threshold_var is not None:
            selector = VarianceThreshold(threshold=threshold_var)
            selector.fit(X)
            cols_var = X.columns[selector.get_support()]
            removed_count = initial_count - len(cols_var)
            X = X[cols_var]
            self._log(f"Variância: {removed_count} features removidas. Restantes: {X.shape[1]}")
        else:
            self._log("Remoção de Variância: Pular.")

        # remoção por correlação de pearson 
        if threshold_corr is not None:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
            X = X.drop(columns=to_drop)
            self._log(f"Correlação (>{threshold_corr}): {len(to_drop)} features redundantes removidas. Restantes: {X.shape[1]}")
        else:
            self._log("Remover Correlação: Pular.")

        # random forest importance
        if top_n_features is not None:
            if X.shape[1] > top_n_features:
                rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=X.columns)
                selected_feats = importances.nlargest(top_n_features).index.tolist()
                X = X[selected_feats]
                self._log(f"Random Forest: Top {top_n_features} selecionadas.")
            else:
                self._log("Random Forest: Ignorado (Features atuais <= Top N).")
        else:
            self._log("Random Forest: Pular.")

        self._log(f"Features Finais ({X.shape[1]}) - {X.columns.tolist()}")
        self._log("--- Fim do Processo de Seleção de Features ---\n")

        return X

    def _normalize_data(self, X, method=None): 
        match method:
            case "MinMaxScaler":
                scaler = MinMaxScaler()
            case "StandardScaler":
                scaler = StandardScaler()
            case "RobustScaler":
                scaler = RobustScaler()
            case _:
                self._log("Normalização: Dados originais mantidos.")
                return X.values if hasattr(X, 'values') else X

        # aplica a transformação se encontrou um método válido
        scaled_x = scaler.fit_transform(X)
        self._log(f"Normalização: {method}")
        return scaled_x

    def _handle_missing_values(self, X, method='0'):
        match str(method).lower():
            case 'media':
                self._log("Tratamento de Nulos: Preenchendo com a MÉDIA das colunas...")
                return X.fillna(X.mean())
            case 'mediana':
                self._log("Tratamento de Nulos: Preenchendo com a MEDIANA das colunas...")
                return X.fillna(X.median())
            case 'moda':
                self._log("Tratamento de Nulos: Preenchendo com a MODA das colunas...")
                return X.fillna(X.mode().iloc[0])
            case '0':
                self._log("Tratamento de Nulos: Preenchendo com ZERO.")
                return X.fillna(0)
            case _:
                self._log(f"Aviso: Método de preenchimento '{method}' desconhecido. Usando ZERO por padrão.")
                return X.fillna(0)

    def create_stream(self, df, target_label_col='Label', binary_label=True, 
                      normalize_method=None, threshold_var=None,
                      threshold_corr=None, top_n_features=None,
                      return_stream=True, extra_ignore_cols=None,
                      imputation_method='0'):

        # limpeza básica
        self._log("Limpeza: Removendo espaços, identificadores (Flow ID, Timestamp, Unnamed: 0) e colunas vazias...")
        df.columns = df.columns.str.strip()
        target_label_col = target_label_col.strip()
        
        ignore_cols = ['Flow ID', 'Timestamp', 'SimillarHTTP', 'Unnamed: 0']
        
        if extra_ignore_cols:
            if isinstance(extra_ignore_cols, str):
                ignore_cols.append(extra_ignore_cols)
            else:
                ignore_cols.extend(extra_ignore_cols)
                
        cols_to_drop = [c for c in ignore_cols if c in df.columns]
        
        X = df.drop(columns=[target_label_col] + cols_to_drop, errors='ignore')
        
        # tratamento numérico
        self._log("Pré-processamento: Convertendo infinitos...")
        X = X.select_dtypes(include=[np.number])
        X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)
        X = self._handle_missing_values(X, method=imputation_method)

        # normalização
        if normalize_method:
            temp_col_names = X.columns
            temp_x_array = self._normalize_data(X, method=normalize_method)
            X = pd.DataFrame(temp_x_array, columns=temp_col_names)

        # definição do target (y)
        type_lbl = "Binário (0=Normal, 1=Attack)" if binary_label else "Multiclasse"
        self._log(f"Target: Processando coluna '{target_label_col}' como {type_lbl}...")
        
        if binary_label:
            is_benign = df[target_label_col].astype(str).str.strip().str.upper() == 'BENIGN'
            y = np.where(is_benign, 0, 1).astype(np.int8)
            target_names = ['Normal', 'Attack'] 
        else:
            le = LabelEncoder()
            y = le.fit_transform(df[target_label_col].astype(str))
            target_names = le.classes_.tolist()

        # redução da dimensionalidade
        if threshold_var is not None or threshold_corr is not None or top_n_features is not None:
            self._log("Seleção de Features: Iniciando pipeline de redução de dimensionalidade...")
            X = self._remove_features(X, y, 
                                      threshold_var=threshold_var,
                                      threshold_corr=threshold_corr,
                                      top_n_features=top_n_features)
        else:
            self._log("Seleção de Features: Nenhuma técnica selecionada. Mantendo todas as colunas.")

        # extrai dados finais para retorno
        feature_names = X.columns.tolist()
        final_x_array = X.values

        # criação do retorno
        if return_stream:
            self._log("Finalização: Criando objeto NumpyStream para o CapyMOA.\n")
            stream_obj = NumpyStream(
                final_x_array, 
                y, 
                target_name="Class", 
                feature_names=feature_names,
                target_type="categorical"
            )
            return stream_obj, target_names, feature_names
        else:
            self._log("Finalização: Retornando DataFrame pandas processado.\n")
            final_df = pd.DataFrame(final_x_array, columns=feature_names)
            return final_df, y, target_names