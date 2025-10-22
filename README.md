# Detecção de Anomalias em Fluxos de Redes com Online Machine Learning (OML)

Este estudo concentra-se na aplicação de técnicas de Online Machine Learning (OML) no contexto da Mineração de Fluxo de Dados (Data Stream Mining) para a detecção em tempo real de anomalias no fluxo de redes. O objetivo é demonstrar a capacidade de modelos de se adaptarem continuamente a dados não estacionários (conceito de Concept Drift). O ambiente de estudos foi preparado para ser iniciado de duas maneiras: via Docker, ou localmente no VS Code.

## Configuração de Ambiente no VS Code

Caso prefira rodar o projeto localmente, siga estes passos para configurar seu ambiente virtual e instalar as dependências:

-   **Pré-requisito:** Certifique-se de que o **Python 3.10** (ou versão superior) esteja instalado e acessível no PATH do seu sistema.
-   **Clonar:** Clone o repositório para sua máquina.
-   **Abrir Projeto:** Abra a pasta raiz do repositório no **Visual Studio Code**.
-   **Criar Ambiente:**
    -   Pressione **`F1`** (ou `Ctrl/Cmd + Shift + P`) para abrir a Paleta de Comandos.
    -   Digite e selecione: **`Python: Create Environment`**.
-   **Instalação Rápida:**
    -   Escolha seu gerenciador de ambiente (**Venv** ou **Conda**).
    -   Selecione **`Quick Create`**. O VS Code fará o restante, instalando todas as bibliotecas listadas no `requirements.txt` no ambiente isolado.

## Configuração de Ambiente no Docker

Para garantir que todos os artefatos de código, bibliotecas e dependências sejam executados de forma idêntica em qualquer máquina, utilizamos o Docker para contêinerizar e versionar o ambiente, facilitando a reprodução fiel dos estudos e eliminando problemas de compatibilidade.

### 1 Processo de Inicialização 

Antes de tudo, garanta que o **Docker Desktop esteja instalado e rodando**. Em seguida, clone o repositório e navegue até a pasta raiz do projeto.

#### 1.1 Construir a Imagem (Primeiro Uso ou Alteração de Bibliotecas)

Este comando cria o ambiente isolado `oml-study-env` a partir do seu `Dockerfile`.

```bash
docker build -t oml-study-env .
```

#### 1.2 Iniciar o Contêiner

O contêiner deve ser iniciado antes de cada sessão de trabalho. Ele mapeia sua pasta local e a porta 8888.

```bash
docker run -d -p 8888:8888 --name oml-container -v "$(pwd):/app" oml-study-env
```

#### 1.3 Entrar no Contêiner

Você precisa acessar o terminal interno do contêiner para executar o servidor Jupyter.

```bash
docker exec -it oml-container bash
```

#### 1.4 Iniciar o Servidor Jupyter Notebook

Execute este comando dentro do contêiner (root@...:/app#).

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### 1.5 Conexão ao Notebook

Cole no Navegador: Acesse o link no seu navegador. Se o 0.0.0.0 não funcionar, use http://127.0.0.1:8888/ seguido do token. Seu notebook main.ipynb estará disponível e rodando no ambiente Docker isolado.

### 2. Comandos de gerenciamento do Docker

Use estes comandos para controlar o ciclo de vida do ambiente.

- **Parar o Contêiner:** `docker stop oml-container`
- **Remover o Contêiner (se for reiniciar):** `docker rm oml-container`
- **Verificar Contêineres Ativos:** `docker ps`
- **Reconstruir a Imagem:** `docker build -t oml-study-env .`

