import os
import sys
import pandas as pd
import tensorflow as tf
#from data_transformation import get_dataset_partitions
#from src.data_transformation import get_dataset_partitions
#from data_transformation import data_manupulation
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts/train')
    test_data_path:str = os.path.join("artifacts/test")
    data_path:str = os.path.join('artifacts/data')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory("C:/Users/Vijay Rama Raju U/datasets/potato_diease/PlantVillage")
            return dataset
        except Exception as e:
            raise e




