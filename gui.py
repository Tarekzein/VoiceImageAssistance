import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QLineEdit, QMessageBox, QProgressBar, QInputDialog,
                            QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from voice_image_assistant import VoiceImageAssistant
from database import Database

class VoiceThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, assistant, mode='train'):
        super().__init__()
        self.assistant = assistant
        self.mode = mode
        
    def run(self):
        try:
            if self.mode == 'train':
                self.assistant.train_voice_model()
                self.finished.emit("Voice model trained successfully!")
            elif self.mode == 'record':
                audio_data = self.assistant.record_voice()
                self.finished.emit("Voice recorded successfully!")
            elif self.mode == 'verify':
                audio_data = self.assistant.record_voice()
                if self.assistant.verify_voice(audio_data):
                    self.finished.emit("Voice verified successfully!")
                else:
                    self.error.emit("Voice verification failed!")
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.assistant = VoiceImageAssistant()
        self.db = Database()
        self.current_image = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Voice-Image Assistant')
        self.setMinimumSize(1000, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel('Status: Not trained')
        layout.addWidget(self.status_label)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Training group
        training_group = QGroupBox("Voice Training")
        training_layout = QHBoxLayout()
        self.train_button = QPushButton('Train Voice Model')
        self.train_button.clicked.connect(self.train_voice)
        training_layout.addWidget(self.train_button)
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Image addition group
        image_group = QGroupBox("Add New Image")
        image_layout = QVBoxLayout()
        
        # Image selection
        image_select_layout = QHBoxLayout()
        self.select_image_button = QPushButton('Select Image')
        self.select_image_button.clicked.connect(self.select_image)
        image_select_layout.addWidget(self.select_image_button)
        image_layout.addLayout(image_select_layout)
        
        # Voice recording
        voice_layout = QHBoxLayout()
        self.record_voice_button = QPushButton('Record Voice Label')
        self.record_voice_button.clicked.connect(self.record_voice_label)
        self.record_voice_button.setEnabled(False)
        voice_layout.addWidget(self.record_voice_button)
        image_layout.addLayout(voice_layout)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # Recognition group
        recognition_group = QGroupBox("Voice Recognition")
        recognition_layout = QHBoxLayout()
        self.recognize_button = QPushButton('Recognize Voice')
        self.recognize_button.clicked.connect(self.recognize_voice)
        recognition_layout.addWidget(self.recognize_button)
        recognition_group.setLayout(recognition_layout)
        layout.addWidget(recognition_group)
        
        # Image list
        self.image_list_label = QLabel('Added Images:')
        layout.addWidget(self.image_list_label)
        
        self.update_image_list()
        
    def train_voice(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Record multiple samples for training
        try:
            QMessageBox.information(self, 'Training', 
                                  'Please provide 5 voice samples for better recognition.\n'
                                  'Speak clearly and consistently for each sample.')
            
            # Get the label for training
            label, ok = QInputDialog.getText(self, 'Training Label', 
                                           'Enter the word you will speak for training:')
            if not ok or not label:
                raise ValueError("No label provided for training")
            
            for i in range(5):  # Record 5 samples
                QMessageBox.information(self, 'Training', 
                                      f'Recording sample {i+1}/5\n'
                                      f'Please speak the word: "{label}"')
                audio_data = self.assistant.record_voice()
                self.assistant.add_voice_sample(audio_data, label)  # Use the actual label
            
            self.assistant.train_voice_model()
            self.status_label.setText('Status: Voice model trained successfully!')
            QMessageBox.information(self, 'Success', 
                                  'Voice model trained successfully!\n'
                                  'You can now add images and test voice recognition.')
        except Exception as e:
            self.on_error(str(e))
        finally:
            self.progress_bar.setVisible(False)
        
    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f'Status: Error - {error_message}')
        QMessageBox.critical(self, 'Error', error_message)
        
    def select_image(self):
        if not self.assistant.is_trained:
            QMessageBox.warning(self, 'Warning', 'Please train your voice model first!')
            return
            
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 
                                                 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            self.current_image = cv2.imread(file_name)
            if self.current_image is not None:
                self.display_image(self.current_image)
                self.record_voice_button.setEnabled(True)
                self.status_label.setText('Status: Image loaded. Please record voice label.')
            else:
                QMessageBox.warning(self, 'Error', 'Failed to load image!')
                
    def record_voice_label(self):
        if self.current_image is None:
            QMessageBox.warning(self, 'Warning', 'Please select an image first!')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            # Get the label first
            label, ok = QInputDialog.getText(self, 'Image Label', 
                                           'Enter a label for this image:')
            if not ok or not label:
                raise ValueError("No label provided")
                
            QMessageBox.information(self, 'Recording', 
                                  f'Please speak the word: "{label}"\n'
                                  'Speak clearly and use the same voice as during training.')
            audio_data = self.assistant.record_voice()
            self.assistant.add_voice_sample(audio_data, label)  # Use the actual label
            
            # Save the image with the label
            try:
                self.db.save_image(label, self.current_image)
                self.update_image_list()
                QMessageBox.information(self, 'Success', f'Image "{label}" added successfully!')
                self.current_image = None
                self.record_voice_button.setEnabled(False)
                self.image_label.clear()
            except Exception as e:
                self.on_error(str(e))
                
        except Exception as e:
            self.on_error(str(e))
        finally:
            self.progress_bar.setVisible(False)
            
    def recognize_voice(self):
        if not self.assistant.is_trained:
            QMessageBox.warning(self, 'Warning', 'Please train your voice model first!')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            QMessageBox.information(self, 'Recognition', 
                                  'Please speak the label of the image you want to see.\n'
                                  'Speak clearly and use the same voice as during training.')
            audio_data = self.assistant.record_voice()
            label = self.assistant.verify_voice(audio_data)
            
            if label:
                print(f"Recognized label: {label}")  # Debug output
                self.show_image(label)
            else:
                QMessageBox.warning(self, 'Warning', 
                                  'Voice not recognized. Please try again.\n'
                                  'Make sure to speak clearly and use the same voice as during training.')
        except Exception as e:
            self.on_error(str(e))
        finally:
            self.progress_bar.setVisible(False)
        
    def show_image(self, label):
        image = self.db.get_image(label)
        if image is not None:
            self.display_image(image)
        else:
            QMessageBox.warning(self, 'Warning', f'No image found for label: {label}')
            
    def display_image(self, image):
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                               Qt.AspectRatioMode.KeepAspectRatio))
        
    def update_image_list(self):
        images = self.db.get_all_images()
        image_list = [img.label for img in images]
        self.image_list_label.setText(f'Added Images: {", ".join(image_list)}')
        
    def closeEvent(self, event):
        self.db.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 