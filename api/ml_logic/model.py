import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from datetime import datetime

def define_model():
    '''
    The VGG16 is defined
    '''

    # Load the VGG-16 model (pre-trained on ImageNet)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the convolutional layers
    for layer in base_model.layers:
    	layer.trainable = False

    # Add custom layers for your specific task (e.g., fine-tuning for a different dataset)
    x = Flatten()(base_model.output)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='relu')(x)
    output = Dense(5, activation='softmax')(x)  # Replace num_classes with your number of classes

    # Create the custom model
    model = Model(inputs=base_model.input, outputs=output)

    return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):

    es = EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True,verbose=1)

    history = model.fit(
	    X_train,
	    y_train,
	    validation_data = (X_val, y_val),
	    epochs = 300,
	    batch_size = 16,
	    callbacks=[es],
	    verbose=1
    )

    return history

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test)

def predict_model(model, X_new):
    return model.predict(X_new)

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f'model_{timestamp}.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    file.close()
