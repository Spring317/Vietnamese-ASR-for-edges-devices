import librosa
from torch import Tensor
from transformers import Wav2Vec2Processor
from numpy import ndarray


class Encoder:
    """This class preprocesses audio data for the Wav2Vec2 model
    ----------------------------------------------------------------
    Args:
        chunk (ndarray): audio chunk recorded from device
        model_id (str): The model identifier from huggingface to load the processor
        org_sr (int): The original sampling rate of the audio file
        target_sr (int): The target sampling rate for the model
    return:
        inputs (torch.Tensor): The input tensor for the Wav2Vec2 model
    """
    def __init__(self, chunk: ndarray, org_sr: int =44100, target_sr: int = 16000, model_id: str = "khanhld/wav2vec2-base-vietnamese-160h") -> None:
        self.__chunk = chunk
        self.__processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.__orig_sr = org_sr
        self.__target_sr = target_sr

    def encode(self) -> Tensor:
        """Load audio and preprocess it for the Wav2Vec2 model via the processor"""
        chunk = librosa.resample(self.__chunk, orig_sr=self.__orig_sr, target_sr=self.__target_sr)
        sampling_rate = self.__target_sr
        inputs = self.__processor(chunk, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values
        return inputs
    
    def decode(self, output: Tensor) -> str:
        """Decode the model prediction into str"""
        text = self.__processor.batch_decode(output)
        return text