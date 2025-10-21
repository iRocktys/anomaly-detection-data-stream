# Usa uma imagem base leve e estável
FROM python:3.10-slim

WORKDIR /app

# Instala Jupyter e utilitários básicos
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl procps locales && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install jupyter notebook

# Instala NUMPY SEPARADAMENTE (para garantir que esteja presente antes de qualquer outra coisa)
# Mantemos essa linha por segurança, caso o base-slim não tenha a versão correta
RUN pip install numpy==1.26.1

# Copia e instala o restante dos requerimentos (river, pandas, sklearn)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]