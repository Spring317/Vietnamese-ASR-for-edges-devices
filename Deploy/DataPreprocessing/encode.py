import librosa
from torch import Tensor
from transformers import Wav2Vec2Processor
import typing

class Encoder:
    """This class preprocesses audio data for the Wav2Vec2 model
    ----------------------------------------------------------------
    Args:
        file_path (str): The path to the audio file
        model_id (str): The model identifier from huggingface to load the processor

    return:
        inputs (torch.Tensor): The input tensor for the Wav2Vec2 model
    """
    def __init__(self, file_path: str, model_id: str = "khanhld/wav2vec2-base-vietnamese-160h") -> None:
        self.__file_path = file_path
        self.__processor = Wav2Vec2Processor.from_pretrained(model_id)

    def encode(self) -> Tensor:
        """Load audio and preprocess it for the Wav2Vec2 model via the processor"""
        
        speech_array, sampling_rate = librosa.load(self.__file_path, sr=16000)
        inputs = self.__processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values
        return inputs
    
    def decode(self, output: Tensor) -> str:
        """Decode the model prediction into str"""
        text = self.__processor.batch_decode(output)
        return text