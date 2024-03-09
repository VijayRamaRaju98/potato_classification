import os
import sys

import numpy as np
from src.components.data_ingestion import DataIngestion


import tensorflow as tf
#sys.path.insert(1,"C:\\Users\\Vijay Rama Raju U\\datasets\\vijay\\potato\\src\\components")
#from src.components.data_ingestion import DataIngestion

class get_dataset_partitions:
    def __init__(self):
        name = "hello world"

    def train_test_val_split(dataset, train_split=0.8, test_split=0.1, val_split=0.1, shuffle=True, shuffle_size=1000):
        try:

            dataset_size = len(dataset)
            if shuffle:
                dataset = dataset.shuffle(shuffle_size, seed=12)
                train_size = int(train_split * dataset_size)
                val_size = int(val_split*dataset_size)

                train_ds = dataset.take(train_size)
                val_ds = dataset.skip(train_size).take(val_size)
                test_ds = dataset.skip(train_size).skip(val_size)

                return train_ds, test_ds, val_ds
        except Exception as e:
            raise e    

class data_manupulation:
    def __init__(self):
        number = 10

    def shuffle_the_data(self,train_ds, test_ds,val_ds):
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_ds, test_ds, val_ds
    


    def model_building(self):
        input_shape_ = (32,256,256,3)
        n_clases = 3
        data_aug = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])
        
        data_resizing = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(256,256),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ])


        model = tf.keras.models.Sequential([
            data_aug,
            data_resizing,
            
            tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape_),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(n_clases,activation='softmax')

        ])

        model.build(input_shape=input_shape_)
        print(model.summary())
        model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        
        return model
    


if __name__=="__main__":
    ingestion = DataIngestion()
    dataset = ingestion.initiate_data_ingestion()
    train_data,test_data, val_data = get_dataset_partitions.train_test_val_split(dataset)
    train_ds, test_ds, val_ds = data_manupulation.shuffle_the_data(self=data_manupulation,train_ds=train_data, test_ds=test_data,val_ds=val_data)
    model = data_manupulation.model_building(self=data_manupulation)
    model_training = model.fit(train_ds, batch_size=32,validation_data=val_ds, verbose=1,epochs=2)
    print(model_training.history['accuracy'])