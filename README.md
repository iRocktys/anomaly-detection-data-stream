# anomaly-detection-data-stream

# Guia de Acesso e Gerenciamento do Ambiente Docker

Este documento explica como iniciar e conectar-se ao ambiente isolado `oml-study-env`, que contém todas as bibliotecas necessárias (`river`, `pandas`, `scikit-learn`).

## 1. Processo de Inicialização

Assuma que você está no terminal na pasta raiz do projeto.

### Construir a Imagem (Primeiro Uso ou Alteração de Bibliotecas)

Este comando cria o ambiente isolado `oml-study-env` a partir do seu `Dockerfile`.

```bash
docker build -t oml-study-env .
```

### 1.2 Iniciar o Contêiner

O contêiner deve ser iniciado antes de cada sessão de trabalho. Ele mapeia sua pasta local e a porta 8888.

```bash
docker run -d -p 8888:8888 --name oml-container -v "$(pwd):/app" oml-study-env
```

### 1.3 Entrar no Contêiner

Você precisa acessar o terminal interno do contêiner para executar o servidor Jupyter.

```bash
docker exec -it oml-container bash
```

### 1.4 Iniciar o Servidor Jupyter Notebook

Execute este comando dentro do contêiner (root@...:/app#).

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 1.5 Conexão ao Notebook

Cole no Navegador: Acesse o link no seu navegador. Se o 0.0.0.0 não funcionar, use http://127.0.0.1:8888/ seguido do token. Seu notebook main.ipynb estará disponível e rodando no ambiente Docker isolado.

## 2. Comandos de gerenciamento do Docker

Use estes comandos para controlar o ciclo de vida do ambiente.

- **Parar o Contêiner:** `docker stop oml-container`
- **Remover o Contêiner (se for reiniciar):** `docker rm oml-container`
- **Verificar Contêineres Ativos:** `docker ps`
- **Reconstruir a Imagem:** `docker build -t oml-study-env .`

