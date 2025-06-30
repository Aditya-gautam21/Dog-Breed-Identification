from data_loader import get_data_generators
from model import build_model

def train_model(data_dir='data/train', epochs=10, batch_size=32):
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)
    model = build_model(input_shape=(224, 224, 3), num_classes=train_gen.num_classes)
    
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save("dog_breed_model.h5")
    print("✅ Model training complete and saved.")

if __name__ == "__main__":
    train_model()
