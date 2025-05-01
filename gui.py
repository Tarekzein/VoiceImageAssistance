import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QLineEdit, QMessageBox, QProgressBar, QInputDialog)
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
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Voice-Image Assistant')
        self.setMinimumSize(800, 600)
        
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
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 2px solid #ccc;")
        layout.addWidget(self.image_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton('Train Voice Model')
        self.train_button.clicked.connect(self.train_voice)
        button_layout.addWidget(self.train_button)
        
        self.add_image_button = QPushButton('Add Image')
        self.add_image_button.clicked.connect(self.add_image)
        button_layout.addWidget(self.add_image_button)
        
        self.recognize_button = QPushButton('Recognize Voice')
        self.recognize_button.clicked.connect(self.recognize_voice)
        button_layout.addWidget(self.recognize_button)
        
        layout.addLayout(button_layout)
        
        # Image list
        self.image_list_label = QLabel('Added Images:')
        layout.addWidget(self.image_list_label)
        
        self.update_image_list()
        
    def train_voice(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.voice_thread = VoiceThread(self.assistant, 'train')
        self.voice_thread.finished.connect(self.on_training_finished)
        self.voice_thread.error.connect(self.on_error)
        self.voice_thread.start()
        
    def on_training_finished(self, message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f'Status: {message}')
        QMessageBox.information(self, 'Success', message)
        
    def on_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f'Status: Error - {error_message}')
        QMessageBox.critical(self, 'Error', error_message)
        
    def add_image(self):
        if not self.assistant.is_trained:
            QMessageBox.warning(self, 'Warning', 'Please train your voice model first!')
            return
            
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 
                                                 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            label, ok = QInputDialog.getText(self, 'Image Label', 
                                           'Enter a label for this image:')
            if ok and label:
                self.voice_thread = VoiceThread(self.assistant, 'verify')
                self.voice_thread.finished.connect(lambda: self.save_image(file_name, label))
                self.voice_thread.error.connect(self.on_error)
                self.voice_thread.start()
                
    def save_image(self, file_name, label):
        try:
            image = cv2.imread(file_name)
            _, buffer = cv2.imencode('.png', image)
            self.db.save_image(label, buffer.tobytes())
            self.update_image_list()
            QMessageBox.information(self, 'Success', f'Image "{label}" added successfully!')
        except Exception as e:
            self.on_error(str(e))
            
    def recognize_voice(self):
        if not self.assistant.is_trained:
            QMessageBox.warning(self, 'Warning', 'Please train your voice model first!')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.voice_thread = VoiceThread(self.assistant, 'verify')
        self.voice_thread.finished.connect(self.show_image)
        self.voice_thread.error.connect(self.on_error)
        self.voice_thread.start()
        
    def show_image(self, label):
        self.progress_bar.setVisible(False)
        image_data = self.db.get_image(label)
        if image_data:
            nparr = np.frombuffer(image_data.image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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