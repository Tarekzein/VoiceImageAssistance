# Voice-Image Assistant

This application allows you to create a personalized voice-controlled image recognition system. It can recognize your voice and associate it with images you provide, creating a secure and personalized experience.

## Features

- Voice authentication to ensure only your voice can access the system
- Image storage and retrieval using voice commands
- Voice model training and persistence
- Modern GUI interface
- SQLite database for data persistence
- Multi-threaded voice processing

## Requirements

- Python 3.8 or higher
- Microphone
- Webcam (optional, for testing)

## Installation

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have a working microphone connected to your system.

## Usage

1. Run the application:
```bash
python gui.py
```

2. Using the GUI:
   - Click "Train Voice Model" to start voice training (required before using other features)
   - Use "Add Image" to add new images to the database
   - Click "Recognize Voice" to use voice commands to retrieve images
   - The status bar shows the current state of the application
   - The image list shows all added images
   - Progress bar indicates ongoing operations

## How it Works

1. **Voice Training**: 
   - The system records multiple samples of your voice
   - Creates a unique voice profile using MFCC features
   - Stores the voice model in the database

2. **Image Management**: 
   - Add images through the GUI
   - Each image requires:
     - Image file selection
     - Label assignment
     - Voice verification
   - Images are stored in the SQLite database

3. **Voice Recognition**: 
   - Speak the label of an image
   - System verifies your voice
   - Displays the corresponding image if found

## Security Features

- Voice authentication ensures only your voice can access the system
- Voice verification is required for adding new images
- The system uses MFCC (Mel-frequency cepstral coefficients) for voice feature extraction
- All data is stored locally in an SQLite database

## Technical Details

- Built with PyQt6 for the GUI
- Uses SQLAlchemy for database operations
- Implements multi-threading for voice processing
- Stores images and voice models in SQLite database
- Uses OpenCV for image processing

## Notes

- Make sure to speak clearly when training the voice model
- The system works best in a quiet environment
- You can adjust the voice verification threshold in the code if needed
- The database file (voice_image_assistant.db) is created automatically
- All data persists between sessions 