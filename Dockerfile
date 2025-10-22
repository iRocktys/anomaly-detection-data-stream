FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl procps locales && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install jupyter notebook

RUN pip install numpy==1.26.1

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]