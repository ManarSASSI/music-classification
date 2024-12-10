from flask_cors import CORS
import librosa
import numpy as np
import joblib
from flask import Flask, request, jsonify
import io
import soundfile as sf
import os

app = Flask(__name__)
CORS(app) 
# Liste des genres de musique
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Charger le modèle SVM avec un chemin absolu
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '/app/models/svm_model.pkl')

try:
    model = joblib.load(model_path)
    print("Modèle SVM chargé avec succès.")
except FileNotFoundError:
    print(f"Erreur : Le fichier svm_model.pkl est introuvable au chemin {model_path}")
    raise
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier audio fourni."}), 400

    file = request.files['file']

    try:
        # Lire l'audio
        audio_data, sr = sf.read(io.BytesIO(file.read()), dtype='float32')

        # Ajouter du padding si nécessaire
        min_duration = 30.0
        target_length = int(min_duration * sr)
        if len(audio_data) < target_length:
            pad_length = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_length), mode='constant')

        # Calcul du mel-spectrogramme
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Aplatir et ajuster la taille
        mel_spec_flat = mel_spec_db.flatten()

        target_features = 100
        if len(mel_spec_flat) > target_features:
            mel_spec_flat = mel_spec_flat[:target_features]
        else:
            mel_spec_flat = np.pad(mel_spec_flat, (0, target_features - len(mel_spec_flat)), mode='constant')

        mel_spec_flat = mel_spec_flat.reshape(1, -1)

        # Ajouter des logs pour les caractéristiques
        print("Caractéristiques extraites : ", mel_spec_flat)

        # Prédiction
        genre_pred = model.predict(mel_spec_flat)
        genre_pred_int = int(genre_pred[0])

        # Retourner le nom du genre
        genre_name = classes[genre_pred_int]

        return jsonify({'predicted_genre': genre_name}), 200
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
