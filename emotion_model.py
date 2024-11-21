from tensorflow.keras.models import load_model

def load_emotion_model():
    # Load a pre-trained model (example: FER-2013-based model)
    model = load_model('emotion_model.h5')  # Replace with your model path
    return model
