from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import base64
import cv2
import numpy as np

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    label = Column(String, unique=True, nullable=False)
    image_data = Column(Text, nullable=False)  # Store base64 string
    created_at = Column(DateTime, default=datetime.utcnow)
    
class VoiceModel(Base):
    __tablename__ = 'voice_models'
    
    id = Column(Integer, primary_key=True)
    features = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Database:
    def __init__(self, db_path='voice_image_assistant.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_image(self, label, image_data):
        """Save an image to the database in base64 format"""
        # Convert image to base64
        if isinstance(image_data, np.ndarray):
            _, buffer = cv2.imencode('.png', image_data)
            base64_image = base64.b64encode(buffer).decode('utf-8')
        else:
            base64_image = image_data
            
        image = Image(label=label, image_data=base64_image)
        self.session.add(image)
        self.session.commit()
        return image
    
    def get_image(self, label):
        """Retrieve an image from the database"""
        image_record = self.session.query(Image).filter_by(label=label).first()
        if image_record:
            # Convert base64 back to image
            image_data = base64.b64decode(image_record.image_data)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None
    
    def get_all_images(self):
        """Retrieve all images from the database"""
        return self.session.query(Image).all()
    
    def save_voice_model(self, features):
        """Save voice model features to the database"""
        voice_model = VoiceModel(features=features)
        self.session.add(voice_model)
        self.session.commit()
        return voice_model
    
    def get_latest_voice_model(self):
        """Retrieve the latest voice model from the database"""
        return self.session.query(VoiceModel).order_by(VoiceModel.created_at.desc()).first()
    
    def delete_image(self, label):
        """Delete an image from the database"""
        image = self.session.query(Image).filter_by(label=label).first()
        if image:
            self.session.delete(image)
            self.session.commit()
            return True
        return False
    
    def close(self):
        """Close the database session"""
        self.session.close() 