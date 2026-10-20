import pandas as pd


# Organiza os nomes e motivos das características excluídas em cada etapa.
def buildFeatureRemovalReport(records):
    return pd.DataFrame(records, columns=["Stage", "Feature", "Reason"])


# Exibe todas as características efetivamente removidas sem truncar os nomes.
def printFeatureRemovalReport(report, title="Remoção de features"):
    print(title)
    if report.empty:
        print("Nenhuma feature removida.")
    else:
        print(report.to_string(index=False))
