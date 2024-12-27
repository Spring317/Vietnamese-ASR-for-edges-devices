import torch
from torch.ao.quantization import quantize_dynamic
from optimum.quanto import quantize, qint8
import tvm
from tvm import relay, auto_scheduler
from transformers import Wav2Vec2Model, Wav2Vec2Processor, QuantoConfig
import numpy as np
from tvm.contrib import graph_executor
from tvm.driver import tvmc
import librosa
import json
from typing import List


class ModelWrapper(torch.nn.Module):
    """
    This class returns the last hidden state of the model
    -----------------------------------------------------
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
    ---------------------------------------------------------------
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
    """"
    Quantize model to int8
    Hyperparameters:
        model_name: str: the model name
    Return:
        torch.nn.Module: the quantized model
    """    
        
    def __init__(self, model_name: str) -> torch.nn.Module:
        quantize_config = QuantoConfig(weights="int8")
        self.__model = Wav2Vec2Model.from_pretrained(model_name, quantization_config=quantize_config)
        
    def quantize(self) -> torch.nn.Module:    
        # quantized_model = quantize_dynamic(self.__model, {torch.nn.Linear}, dtype=torch.qint8)
        return self.__model
    
    # def quanto_optims(self):
        r

class Deployment:
    """Deploy the model to llvm by using tvm
    ----------------------------------------------------
    Hyperparameters:
        model: torch.nn.Module: the model to be deployed
        target: str: the target of the deployment
    Return:
        tvm.relay: the relay model
    """

    def __init__(self, model: torch.nn.Module, target: str) -> tvm.relay:
        self.__model = model
        self.__target = target
    
    def deploy(self, inputs: List) -> tvm.relay:
        """This method deploy model to llvm using tvm
        -------------------------------------------------
        Parameters:
            inputs: List: the input of the model
        Return:
            tvm.relay: the relay model"""

        model = ModelWrapper(self.__model)
        traced_model = torch.jit.trace(model, inputs[0])
        
        input_shape = inputs[0].shape
        shape_list = [(f'input', input_shape)]
        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)
        
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=self.__target, params=params)
        return lib

    def compile(self, lib: tvm.relay) -> graph_executor.GraphModule:
        """This method is used to compile the model
        -------------------------------------------
        Parameters:
            lib: tvm.relay: the relay model
        Return:
            graph_executor.GraphModule: the compiled model
        """
        m = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))
        return m
    
    def run(self, m: graph_executor.GraphModule, inputs: List) -> np.ndarray:
        """This method is used to run the model
        ---------------------------------------
        Parameters:
            m: graph_executor.GraphModule: the compiled model
            inputs: List: the input of the model
        Return:
            np.ndarray: the output of the model
        """
        m.set_input('input0', inputs[0].numpy())
        m.run()
        return m.get_output(0).asnumpy()
    
    
def main():
    model_name = 'khanhld/wav2vec2-base-vietnamese-160h'
    audio = Audio('prompts.txt', model_name)
    audio.get_audio_path()
    inputs = audio.preprocess()

    quantize = Quantize(model_name)
    quantized_model = quantize.quantize()

    deployment = Deployment(quantized_model, 'llvm')
    
    lib = deployment.deploy(inputs)
    m = deployment.compile(lib)
    output = deployment.run(m, inputs)
    print(output) 


if __name__ == '__main__':
    main()

    

        
