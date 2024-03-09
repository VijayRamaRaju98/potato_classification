from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import get_dataset_partitions,data_manupulation




if __name__=="__main__":
    ingestion = DataIngestion()
    dataset = ingestion.initiate_data_ingestion()
    train_data,test_data, val_data = get_dataset_partitions.train_test_val_split(dataset)
    train_ds, test_ds, val_ds = data_manupulation.shuffle_the_data(train_ds=train_data, test_ds=test_data,val_ds=val_data)
    model = data_manupulation.model_building()
    model_training = model.fit(train_ds, batch_size=32,validation_data=val_ds, verbose=1,epochs=1)
    print(model_training.history['accuracy'])