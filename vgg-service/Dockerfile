# Étape 1: Utiliser une image de base avec Python
FROM python:3.9-slim

# Étape 2: Installer les dépendances du projet
RUN pip install --no-cache-dir flask tensorflow librosa flask-cors pillow

# Étape 3: Définir le répertoire de travail dans le conteneur
WORKDIR /app


# Étape 4: Copier le code source dans le conteneur
COPY . /app

# Step 5: Ensure the models directory exists (for the shared volume)
RUN mkdir -p /app/models

# Étape 6: Exposer le port de l'application Flask
EXPOSE 5001

# Étape 7: Démarrer l'application Flask
CMD ["python", "app.py"]