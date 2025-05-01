from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True)
    label = Column(String, unique=True, nullable=False)
    image_data = Column(LargeBinary, nullable=False)
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
        """Save an image to the database"""
        image = Image(label=label, image_data=image_data)
        self.session.add(image)
        self.session.commit()
        return image
    
    def get_image(self, label):
        """Retrieve an image from the database"""
        return self.session.query(Image).filter_by(label=label).first()
    
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