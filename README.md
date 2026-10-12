# Anomaly Detection — arquitetura refatorada

O projeto utiliza notebooks somente como camada de configuração e inspeção. Toda lógica reutilizável fica em `src/`, e todos os experimentos de treinamento são binários: `BENIGN = 0` e `ATTACK = 1`.

## Notebooks na raiz

- `DSPOT.ipynb`: estudo do DSPOT usando a mesma implementação usada no pipeline real.
- `AnomalyDetection.ipynb`: treinamento dos detectores de anomalia.
- `Classification.ipynb`: treinamento prequential dos classificadores binários.
- `Optimization.ipynb`: otimização Optuna de detectores ou classificadores com F1 como objetivo.
- `Scenarios.ipynb`: criação dos cenários janelados e binários.
- `Similarity.ipynb`: análise estatística de semelhança entre ataques por distância de Mahalanobis, sem treinamento de classificação.

## Organização de `src`

- `Anomaly/`: registry dos detectores.
- `Classification/`: registry dos classificadores binários ARF, HAT, HT e Naive Bayes.
- `Data/`: leitura, seleção de features, codificação binária, imputação e normalização incrementais.
- `DSPOT/`: estudo e diagnóstico do DSPOT sem duplicar sua implementação matemática.
- `Experiment/`: configuração e execução comum dos experimentos.
- `Metrics/`: métricas binárias incrementais e acumuladas.
- `Models/`: infraestrutura comum dos registries CapyMOA, sem duplicar criação e validação entre tarefas.
- `Optimization/`: motor Optuna e espaços de busca por modelo/tarefa.
- `Pipeline/`: pipelines de anomaly detection e classification, contratos e persistência de resultados.
- `Plots/`: única camada responsável por implementar os gráficos.
- `Scenario/`: janelamento, agregação e montagem dos cenários.
- `Similarity/`: análise de semelhança entre tipos de ataque.
- `Scores/`: médias móveis dos anomaly scores.
- `Thresholds/`: thresholds fixo e DSPOT.
- `Training/`: estratégias de treinamento dos detectores de anomalia.

## Saídas

Código de plot fica exclusivamente em `src/Plots`. Imagens e resultados nunca são gravados dentro de `src`; os artefatos são organizados sob `output/`:

- `output/Experiments/AnomalyDetection/...`
- `output/Experiments/Classification/...`
- `output/Optimization/...`
- `output/Scenarios/...`
- `output/Similarity/...`
- `output/DSPOT/...`

Cada execução completa salva `instances.csv`, `windows.csv`, `stream_metrics.csv`, `run_metadata.json` e a pasta `plots/`.

## Regra de rótulo

Os datasets usados em treinamento possuem somente `BENIGN` e `ATTACK`. Na criação dos cenários, o tipo real de ataque é mantido no arquivo `*_manifest.csv` para auditoria, mas não é utilizado como classe pelo treinamento.

## Migração da versão anterior

O arquivo `FILES_TO_REMOVE.txt` lista os fontes legados que devem ser removidos ao substituir uma instalação anterior. Também devem ser removidos todos os diretórios `__pycache__` e arquivos `.pyc`. A forma mais segura de atualizar é substituir integralmente a pasta `src/` antiga pela pasta `src/` deste pacote e sobrescrever os notebooks da raiz.
