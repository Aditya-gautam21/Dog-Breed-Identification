import os
import datetime
from data_loader import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def train_model(data_dir='data/train', epochs=10, batch_size=32):
    # Load data
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)

    # Build model
    model = build_model(input_shape=(224, 224, 3), num_classes=train_gen.num_classes)

    # 📁 Callbacks directory
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs("checkpoints", exist_ok=True)

    # ✅ Callbacks
    callbacks = [
        TensorBoard(log_dir=log_dir),
        ModelCheckpoint("checkpoints/best_model.h5", save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]

    # 🏋️ Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # 💾 Save final model
    model.save("dog_breed_model.h5")
    print("✅ Model training complete and saved.")

if __name__ == "__main__":
    train_model()
