# Utiliser une image Python officielle
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the models directory exists (for the shared volume)
RUN mkdir -p /app/models

# Exposer le port sur lequel Flask tourne
EXPOSE 5000

# Démarrer l'application Flask
CMD ["python", "app.py"]
