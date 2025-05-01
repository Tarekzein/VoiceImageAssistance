import os
import cv2
import numpy as np
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from datetime import datetime

class VoiceImageAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.voice_model = None
        self.image_database = {}
        self.voice_features = []
        self.scaler = StandardScaler()
        self.is_trained = False
        self.sample_rate = 44100
        
    def record_voice(self, duration=5):
        """Record voice for training or recognition"""
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate), 
                         samplerate=self.sample_rate, 
                         channels=1, 
                         dtype='float32')
        sd.wait()
        return recording.flatten()
    
    def play_audio(self, audio_data):
        """Play recorded audio"""
        sd.play(audio_data, self.sample_rate)
        sd.wait()
    
    def extract_voice_features(self, audio_data):
        """Extract MFCC features from audio data"""
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, 
                                   sr=self.sample_rate, 
                                   n_mfcc=20,
                                   n_fft=2048,
                                   hop_length=512)
        
        # Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs])
        return np.mean(features, axis=1)
    
    def train_voice_model(self, num_samples=3):
        """Train the voice model with user's voice samples"""
        print("Please provide 3 voice samples for training...")
        features = []
        recordings = []
        
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            print("Please speak now...")
            audio_data = self.record_voice()
            recordings.append(audio_data)
            
            print("Playing back your recording...")
            self.play_audio(audio_data)
            
            features.append(self.extract_voice_features(audio_data))
        
        self.voice_features = np.array(features)
        self.scaler.fit(self.voice_features)
        features_scaled = self.scaler.transform(self.voice_features)
        
        # Create and train the isolation forest model
        self.voice_model = IsolationForest(contamination=0.1, random_state=42)
        self.voice_model.fit(features_scaled)
        
        self.is_trained = True
        print("\nVoice model trained successfully!")
        
        # Save recordings for future reference
        for i, recording in enumerate(recordings):
            sf.write(f'training_sample_{i+1}.wav', recording, self.sample_rate)
    
    def verify_voice(self, audio_data):
        """Verify if the voice matches the trained model"""
        if not self.is_trained:
            return False
        
        features = self.extract_voice_features(audio_data)
        features_scaled = self.scaler.transform([features])
        
        # Get prediction from the model (-1 for anomaly, 1 for normal)
        prediction = self.voice_model.predict(features_scaled)[0]
        
        # Calculate distance to training samples as additional verification
        distances = np.linalg.norm(self.voice_features - features_scaled, axis=1)
        distance_threshold = 0.5
        
        return prediction == 1 and np.mean(distances) < distance_threshold
    
    def add_image(self, image_path, label):
        """Add an image to the database with voice verification"""
        if not os.path.exists(image_path):
            print("Image file not found!")
            return False
        
        print("Please verify your voice to add this image...")
        audio_data = self.record_voice()
        
        if self.verify_voice(audio_data):
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load image!")
                return False
            self.image_database[label] = image
            print(f"Image '{label}' added successfully!")
            return True
        else:
            print("Voice verification failed!")
            return False
    
    def recognize_and_show(self):
        """Recognize voice command and show corresponding image"""
        print("Listening for command...")
        audio_data = self.record_voice()
        
        if not self.verify_voice(audio_data):
            print("Voice verification failed!")
            return
        
        # Convert audio to text
        audio = sr.AudioData(audio_data.tobytes(), self.sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized command: {text}")
            
            if text in self.image_database:
                image = self.image_database[text]
                cv2.imshow(f"Showing: {text}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"No image found for '{text}'")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
    
    def save_model(self, path="voice_model.pkl"):
        """Save the trained model and database"""
        if not self.is_trained:
            print("No model to save!")
            return
        
        data = {
            'voice_features': self.voice_features,
            'scaler': self.scaler,
            'voice_model': self.voice_model,
            'image_database': self.image_database
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path="voice_model.pkl"):
        """Load a trained model and database"""
        if not os.path.exists(path):
            print("No saved model found!")
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.voice_features = data['voice_features']
        self.scaler = data['scaler']
        self.voice_model = data['voice_model']
        self.image_database = data['image_database']
        self.is_trained = True
        print("Model loaded successfully!")

def main():
    assistant = VoiceImageAssistant()
    
    while True:
        print("\nVoice-Image Assistant Menu:")
        print("1. Train voice model")
        print("2. Add new image")
        print("3. Recognize and show image")
        print("4. Save model")
        print("5. Load model")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            assistant.train_voice_model()
        elif choice == "2":
            image_path = input("Enter image path: ")
            label = input("Enter label for the image: ")
            assistant.add_image(image_path, label)
        elif choice == "3":
            assistant.recognize_and_show()
        elif choice == "4":
            assistant.save_model()
        elif choice == "5":
            assistant.load_model()
        elif choice == "6":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 