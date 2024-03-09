
from src.components.data_ingestion import DataIngestion
import tensorflow as tf


class get_dataset_partitions:
    def __init__(self):
        pass

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

    def shuffle_the_data(train_ds, test_ds,val_ds):
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_ds, test_ds, val_ds
    


    def model_building():
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
    


