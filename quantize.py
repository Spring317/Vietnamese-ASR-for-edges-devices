import torch
import tvm
from tvm import relay, auto_scheduler
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
from tvm.contrib import graph_executor
from tvm.driver import tvmc
import librosa
import json
from typing import List


class ModelWrapper(torch.nn.Module):
    """
    This class returns the last hidden state of the model
    Parameters:
        model: torch.nn.Module: the model to be wrapped
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs.last_hidden_state

class Audio:
    """
    This class is used to preprocess audio for preprocessing process
    Parameters: 
        prompt: str: the prompt of the audio
        processor: str: the processor of the audio
    """
    def __init__(self, prompt: str, processor: str) -> None:
        self.__prompt = prompt
        self.__processor = Wav2Vec2Processor.from_pretrained(processor)
    
    def get_audio_path(self) -> None:
        """
        This method is used to get the audio path
        """
        folders = []
        transcripts = []
        filenames = []
        with open(f'vivos/train/{self.__prompt}', 'r') as f:
            for line in f:
                filename, transcript = line.split(' ', 1)
                folder = filename.split('_')[0]
                # print(f"folder: {folder}, transcript: {transcript}, filename: {filename}")
                folders.append(folder)
                transcripts.append(transcript)
                filenames.append(filename)
        
        metadata = {'folders': folders, 'transcripts': transcripts, 'filenames': filenames}
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def preprocess(self) -> List:
        """
        This method is used to preprocess the audio file using the processor from wav2vec2 model
        """
        inputs = []
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        for folder, transcript, filename in zip(metadata['folders'], metadata['transcripts'], metadata['filenames']):
            audio, _ = librosa.load(f'vivos/train/waves/{folder}/{filename}.wav', sr=16000)
            # print(self.__processor(audio, return_tensors='pt', sampling_rate=16000))
            input_values = self.__processor(audio, return_tensors='pt', sampling_rate=16000).input_values

            # if len(input_values.shape) == 2:
            #     input_values = input_values.unsqueeze(0)

            inputs.append(input_values)
        return inputs
    
class Quantize:
    def __init__(self, model: str, target: str, bit_width: int = 8, calibration_samples: int = 1000) -> None:
        self.__model = model
        self.__target = target
        self.__bit_width = bit_width
        self.__calibration_samples = calibration_samples

    def quantize(self, inputs: List) -> None:
        # Load and prepare model
        model = Wav2Vec2Model.from_pretrained(self.__model)
        model.eval()
        model.to('cpu')

        # Verify input shape
        sample_input = inputs[0]
        if not isinstance(sample_input, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(sample_input)}")
            
        print(f"Input tensor shape: {sample_input.shape}")
        
        # Ensure input has the correct dimensions (batch_size, sequence_length)
        # if len(sample_input.shape) != 3:  # Should be [1, 1, sequence_length]
        #     sample_input = sample_input.unsqueeze(0)
            
        # print(f"Adjusted input shape: {sample_input.shape}")

        wrapped_model = ModelWrapper(model)
        
        # Trace with explicit example input
        traced_model = torch.jit.trace(wrapped_model, sample_input)
        
        # Define input shape explicitly
        input_name = 'input'
        shape_list = [(input_name, tuple(sample_input.shape))]
        print(f"Shape list for Relay: {shape_list}")

        # Convert to Relay with explicit shapes
        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        # Print the Relay module for debugging
        print("Relay module:")
        # print(mod)

        # Create target
        target = tvm.target.Target(self.__target)
        
        # Configure quantization
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(calibrate_mode="global_scale",
                                      global_scale=8.0,
                                      nbit_input=self.__bit_width,
                                      nbit_weight=self.__bit_width,
                                      dtype_input="int8",
                                      dtype_weight="int8",
                                      dtype_activation="int8"):
                # Apply quantization
                qmod = relay.quantize.quantize(mod, params)
                
        # Build the quantized model
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(qmod, target=target, params=params)
            
        # Save the quantized model
        lib.export_library("quantized_model.tar")
        print("Quantized model saved as quantized_model.tar")

        # Create graph executor for testing
        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))
        
        # Test inference with properly shaped input
        input_data = sample_input.numpy()
        module.set_input(input_name, input_data)
        module.run()
        output = module.get_output(0)
        print(f"Output shape: {output.shape}")
        
        
    
def main():
    audio = Audio('prompts.txt', 'khanhld/wav2vec2-base-vietnamese-160h')
    audio.get_audio_path()
    inputs = audio.preprocess()

    quantize = Quantize('khanhld/wav2vec2-base-vietnamese-160h','llvm')
    quantize.quantize(inputs)

if __name__ == '__main__':
    main()

    

        
