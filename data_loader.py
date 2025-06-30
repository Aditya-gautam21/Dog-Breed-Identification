import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(train_dir, target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(train_dir,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            subset='training')
    
    val_gen = datagen.flow_from_directory(train_dir,
                                          target_size=target_size,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          subset='validation')
    return train_gen, val_gen
