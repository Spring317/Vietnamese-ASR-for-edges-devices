import torch
from optimum.onnxruntime import ORTModelForCTC
from DataPreprocessing.encode import Encoder


class Prediction:
    """
    Return the prediction of the wav2vec2 model taking the input as the audio file and return the prediction
    -------------------------------------------------------------------------------------------------------------
    Args:
        file_path (str): The path to the audio file
        quantize_path (str): The path to the quantized model
        use_static_quantizer (bool): Whether to use static quantizer or not
        quantize_path_static (str): The path to the model
    """

    def __init__(self, file_path: str, quantize_path: str = 'w2v2_quant', use_static_quantizer: bool = False) -> None:
        self.__file_path = file_path
        self.__quantize_path = quantize_path
        self.__use_static_quantizer = use_static_quantizer
    
    def predict(self):
        """
        return predict the audio file
        """
        if self.__use_static_quantizer:
            model = ORTModelForCTC.from_pretrained(self.__quantize_path + '_statics')
        else:
            model = ORTModelForCTC.from_pretrained(self.__quantize_path) 

        encode = Encoder(self.__file_path)

        input_values = encode.encode()
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcript = encode.decode(pred_ids)

        return pred_transcript

def main():
    file_path = "G:/Vietnamese-ASR-for-edges-devices/Deploy/Test_audio/VIVOSSPK01_R003.wav"
    pred = Prediction(file_path)
    print(pred.predict())

if __name__ == '__main__':
    main()   