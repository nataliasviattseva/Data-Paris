# Utilisez l'image de base Python
FROM python:3.12.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définissez le répertoire de travail dans le conteneur
WORKDIR /Data-Paris

# Copiez le fichier requirements.txt dans le conteneur
COPY requirements.txt /Data-Paris/

# Installez les dépendances du projet
RUN pip install -r requirements.txt

# Copiez le reste des fichiers dans le conteneur
COPY . .

# Exposez le port sur lequel votre application Django s'exécute
EXPOSE 8000

# Définissez la commande pour exécuter votre application Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
