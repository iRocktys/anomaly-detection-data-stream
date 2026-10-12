# Anomaly Detection Data Stream

Projeto organizado com notebooks na raiz e implementação reutilizável em `src/`. O treinamento experimental é exclusivamente binário: `BENIGN/NORMAL = 0` e qualquer ataque = `1`.

## Notebooks

- `DSPOT.ipynb`: estudo e inspeção do funcionamento incremental do DSPOT usando a mesma implementação do treinamento.
- `AnomalyDetection.ipynb`: treinamento e avaliação dos detectores de anomalias.
- `Classification.ipynb`: treinamento e avaliação prequential dos classificadores binários.
- `Optimization.ipynb`: otimização Optuna para detectores ou classificadores.
- `Scenarios.ipynb`: criação do baseline, blocos de ataque e cenários janelados.
- `Similarity.ipynb`: análise estatística de similaridade entre tipos de ataque.

## Organização do backend

- `src/Anomaly`: registry dos detectores.
- `src/Classification`: registry dos classificadores binários.
- `src/Data`: preparação da stream, imputação e normalização incremental.
- `src/DSPOT`: estudo reutilizável do DSPOT.
- `src/Experiment`: configuração e execução dos experimentos.
- `src/Metrics`: métricas binárias incrementais.
- `src/Optimization`: espaços de busca e execução do Optuna.
- `src/Pipeline`: pipelines e persistência dos resultados.
- `src/Plots`: implementação centralizada dos gráficos.
- `src/Scenario`: janelamento e construção dos cenários.
- `src/Scores`: médias móveis de scores.
- `src/Similarity`: análise de similaridade entre ataques.
- `src/Thresholds`: thresholds fixo e DSPOT.
- `src/Training`: estratégias de treinamento incremental.

Os gráficos gerados são persistidos dentro de `output/.../plots`; `src/Plots` contém somente código.

## Compatibilidade

O caminho já usado por `AnomalyDetection.ipynb` foi preservado:

```python
result = ExperimentRunner().run(
    experimentName=experimentName,
    datasets=datasets,
    parameters=parameters,
)
```

`binaryLabel=True` nos datasets e `binary_label=True` em `DataStreamProcessor.create_stream()` continuam aceitos por compatibilidade. Valores `False` são rejeitados porque toda execução experimental é obrigatoriamente binária.

A criação de `LabeledInstance` tenta primeiro `capymoa.instance`, usado pela estrutura original do projeto, e utiliza `capymoa.core` apenas como fallback para versões que exponham a classe nesse módulo.
