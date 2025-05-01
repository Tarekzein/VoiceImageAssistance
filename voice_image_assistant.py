import os
import cv2
import numpy as np
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import librosa
import sounddevice as sd
import soundfile as sf
import pickle
from datetime import datetime

class VoiceImageAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sample_rate = 44100
        self.duration = 3  # Recording duration in seconds
        self.model = None
        self.image_database = {}
        self.is_trained = False
        self.voice_samples = []
        self.labels = []
        self.scaler = StandardScaler()
        self.recognition_threshold = 0.3  # Lower threshold for better recognition
        
    def preprocess_audio(self, audio_data):
        """Preprocess audio data for better feature extraction"""
        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.flatten()
        
        # Remove silence
        audio_data = librosa.effects.trim(audio_data, top_db=20)[0]
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        return emphasized_audio
    
    def extract_features(self, audio_data):
        """Extract enhanced features from audio data"""
        try:
            # Preprocess audio
            audio_data = self.preprocess_audio(audio_data)
            
            # Ensure minimum length
            if len(audio_data) < self.sample_rate:
                audio_data = np.pad(audio_data, (0, max(0, self.sample_rate - len(audio_data))))
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
            pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(delta_mfccs, axis=1),
                np.std(delta_mfccs, axis=1),
                np.array([np.mean(spectral_centroid)]),
                np.array([np.std(spectral_centroid)]),
                np.array([np.mean(spectral_bandwidth)]),
                np.array([np.std(spectral_bandwidth)]),
                np.array([np.mean(spectral_rolloff)]),
                np.array([np.std(spectral_rolloff)]),
                np.array([np.mean(zero_crossing_rate)]),
                np.array([np.std(zero_crossing_rate)]),
                np.array([pitch_mean])
            ])
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Return a default feature vector if extraction fails
            return np.zeros(61)  # 13*2 + 13*2 + 8 + 1 = 61 features
    
    def record_voice(self):
        """Record voice input with validation"""
        print("Recording...")
        try:
            audio_data = sd.rec(int(self.duration * self.sample_rate),
                              samplerate=self.sample_rate,
                              channels=1,
                              dtype='float32')
            sd.wait()
            print("Recording finished")
            
            # Validate recording
            audio_data = audio_data.flatten()
            if len(audio_data) == 0:
                raise ValueError("No audio data recorded")
                
            # Check if audio is too quiet
            if np.max(np.abs(audio_data)) < 0.01:
                raise ValueError("Audio recording is too quiet")
                
            return audio_data
            
        except Exception as e:
            print(f"Error during recording: {e}")
            raise
    
    def train_voice_model(self):
        """Train the voice recognition model with enhanced parameters"""
        if not self.voice_samples:
            raise ValueError("No voice samples available for training")
            
        # Extract features from all samples
        X = []
        valid_labels = []
        
        for i, (sample, label) in enumerate(zip(self.voice_samples, self.labels)):
            try:
                features = self.extract_features(sample)
                X.append(features)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
                
        if not X:
            raise ValueError("No valid features extracted from samples")
            
        X = np.array(X)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Create and train the model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=300,  # More trees for better accuracy
            max_depth=30,      # Deeper trees
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,  # Use all available CPU cores
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X, valid_labels)
        self.is_trained = True
        print("Model trained successfully!")
        
        # Save the model and scaler
        self.save_model()
        
    def verify_voice(self, audio_data):
        """Verify if the voice matches any trained samples with enhanced matching"""
        if not self.is_trained:
            raise ValueError("Model is not trained")
            
        try:
            # Extract features from the new audio
            features = self.extract_features(audio_data)
            features = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            max_prob = np.max(probabilities)
            
            print(f"Recognition confidence: {max_prob:.2f}")  # Debug output
            
            # If the highest probability is above threshold, return the corresponding label
            if max_prob > self.recognition_threshold:
                predicted_label = self.model.classes_[np.argmax(probabilities)]
                return predicted_label
            return None
            
        except Exception as e:
            print(f"Error in voice verification: {e}")
            return None
    
    def add_voice_sample(self, audio_data, label):
        """Add a new voice sample for training"""
        self.voice_samples.append(audio_data)
        self.labels.append(label)
        
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
    
    def save_model(self):
        """Save the trained model and scaler"""
        if not self.is_trained:
            return
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'labels': self.labels
        }
        
        with open('voice_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self):
        """Load a trained model"""
        if os.path.exists('voice_model.pkl'):
            with open('voice_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.labels = model_data['labels']
                self.is_trained = True

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