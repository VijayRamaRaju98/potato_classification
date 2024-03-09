
import tensorflow as tf



class DataIngestion:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        dataset_path = "C:/Users/Vijay Rama Raju U/datasets/potato_diease/PlantVillage"
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(dataset_path)
            global n_classes
            n_classes = len(dataset.class_names)
            
            return dataset
        except Exception as e:
            raise e




