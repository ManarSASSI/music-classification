import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app) 
# Charger le modèle Keras pré-entraîné (assurez-vous que le modèle est dans le même dossier que ce fichier)
model = tf.keras.models.load_model('/app/models/vgg_model.h5')

# Définir les classes des genres musicaux
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Dossier pour stocker les fichiers téléchargés
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configurer Flask pour accepter les fichiers
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Fonction pour vérifier les extensions des fichiers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fonction pour charger et prétraiter le fichier audio
def load_and_preprocess_file(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Calculer le nombre de chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Générer le Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        
        # Redimensionner le spectrogramme
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        
        # Ajouter le spectrogramme à la liste des données
        data.append(mel_spectrogram)
    
    return np.array(data)

# Fonction pour effectuer la prédiction avec le modèle
def model_prediction(X_test):
    Y_pred = model.predict(X_test)
    predicted_categories = np.argmax(Y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si la requête contient un fichier
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Si l'utilisateur n'a pas sélectionné de fichier
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Si le fichier est autorisé
    if file and allowed_file(file.filename):
        # Enregistrer le fichier téléchargé
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Charger et prétraiter le fichier audio
        X_test = load_and_preprocess_file(filepath)

        # Effectuer la prédiction avec le modèle
        class_index = model_prediction(X_test)

        # Retourner la prédiction sous forme de réponse JSON
        return jsonify({'predicted_genre': classes[class_index]}), 200

    else:
        return jsonify({'error': 'File type not allowed. Only .wav files are allowed.'}), 400

if __name__ == '__main__':
    # Démarrer le serveur Flask
  app.run(debug=True, host='0.0.0.0', port=5001)
  