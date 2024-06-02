"""
Created on 31.05.2024 

"""
# Use a pipeline as a high-level helper
from transformers import pipeline
import torchaudio as ta
import torch

# Load model directly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class MIT_AST_model_prob():
    """
    This class is used to classify audio files using the MIT AST model.
    The model is loaded from the Hugging Face model hub.
    
    """
    def __init__(self):
        self.pipe = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        # Use torch.backends.mps.is_built() to check for MPS support
        self.device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
        self.model.to(self.device)  
     

    def classify(self, audio_file):
        sig, fs = ta.load(audio_file)
        sig16 = ta.transforms.Resample(orig_freq=fs, new_freq=16000)(sig[0, :])
        inputs = self.extractor(sig16, sampling_rate=16000, return_tensors="pt")

        # Move inputs to MPS device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1).squeeze()

        # Get the predicted class id and label
        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_class_ids]
        
        # Convert probabilities to a list
        probabilities = probabilities.tolist()

        # Map probabilities to class labels
        id2label = self.model.config.id2label
        class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}

        # Sort class probabilities by probability in descending order
        sorted_class_probabilities = dict(sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True))

        # Keep only the top 5 probabilities
        top_5_class_probabilities = dict(list(sorted_class_probabilities.items())[:5])

        return predicted_label, top_5_class_probabilities

# Example usage:
#model = MIT_AST_model_prob()
# wind_file ='/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_01012024/16/er_file_2024_01_01_14_18_13.wav'
# f_voice_file = '/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_01012024/0/er_file_2024_01_01_11_07_21.wav'
# file = '/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_01012024/0/er_file_2024_01_01_11_07_37.wav'
# speech_file = "/Users/evgenynazarenko/DACS_3_year/Thesis/GardenFiles23/garden_01012024/0/er_file_2024_01_01_11_07_05.wav"
# #label, class_probabilities = model.classify(wind_file)
# print("Predicted label:", label)
# print("Top 5 class/prob:", class_probabilities)
