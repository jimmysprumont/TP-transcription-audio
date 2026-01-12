# Base image avec Python
FROM python:3.12-sli

# Installer ffmpeg (nécessaire pour Whisper)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements (optionnel) et le code
COPY . /app

# Installer les dépendances Python
COPY requirements_env_docker.txt .
RUN pip install --no-cache-dir -r requirements_env_docker.txt

# Exposer le port
EXPOSE 7860

# Lancer FastAPI directement via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

