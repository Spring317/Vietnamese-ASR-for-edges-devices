import torch
from torch.ao.quantization import quantize_dynamic
import tvm
from tvm import relay, auto_scheduler
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, QuantoConfig
import numpy as np
from tvm.contrib import graph_executor
from tvm.driver import tvmc
import librosa
import json
from typing import List, Dict
from onnxruntime.transformers import optimizer
import onnx


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
        
    def __init__(self, model_name: str):
        quantize_config = QuantoConfig(weights="int8")
        # self.__model = Wav2Vec2ForCTC.from_pretrained(model_name, quantization_config=quantize_config)
        self.__model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
    def quantize(self) -> torch.nn.Module:
        """Return quantized_model"""    
        return self.__model
    
    def export_to_onnx(self) -> None:
        """Export model into onnx format"""
        dummy_input = torch.randn(1, 16000)
        input_names = ["audio"]
        output_names = ["text"]
        torch.onnx.export(self.__model, dummy_input, "wav2vec2.onnx", input_names=input_names, output_names=output_names, opset_version=14)

    def optimize_model(self) -> None:
        """Optimize quantized model for cpu deployment"""
        optimized_model = optimizer.optimize_model('wav2vec2.onnx', model_type='wav2vec2', num_heads=12, hidden_size=768)
        optimized_model.save_model_to_file('wav2vec2_optimized.onnx')


class Deployment:
    """Deploy the model to llvm by using tvm
    ----------------------------------------------------
    Hyperparameters:
        model: torch.nn.Module: the model to be deployed
        target: str: the target of the deployment
    Return:
        tvm.relay: the relay model
    """

    def __init__(self, model: torch.nn.Module, target: str) -> None:
        self.__model = model
        self.__target = target
    
    def run_tvm_model(self, mod: tvm.relay, params: Dict, input_name: str, inp: List, target= 'llvm') -> tuple:
        """
        Run pre-quantized model by tvm
        -------------------------------
        Parameters:
            mod: tvm.relay: the relay model
            params: dict: the parameters of the model
            input_name: str: the input name of the model
            inp: List: the input of the model
            target: str: the target of the deployment
        """

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)
            runtime = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))
            runtime.set_input(input_name, inp)
            runtime.run()
            return runtime.get_output(0).asnumpy(), runtime
        
    def deploy(self, inputs: List) -> tvm.relay:
        """
        Deploy the model to llvm
        ------------------------
        Parameters:
            inputs: List: the inputs of the model
        """
        model = ModelWrapper(self.__model)
        # model = self.__model
        model.eval()
        traced_model = torch.jit.trace(model, (inputs[0]))
        input_name=  'input'
        input_shapes = [(input_name, inputs[0].shape)]
        mod, params = relay.frontend.from_pytorch(traced_model, input_shapes)

        return mod, params
    
def main():
    model_name = 'khanhld/wav2vec2-base-vietnamese-160h'
    audio = Audio('prompts.txt', model_name)
    audio.get_audio_path()
    inputs = audio.preprocess()

    quantize = Quantize(model_name)
    quantized_model = quantize.quantize()
    
    #export onnx
    quantize.export_to_onnx()
    quantize.optimize_model()
    
    #load optimized model and give prediction
    # optimized_model = onnx.load('wav2vec2_optimized.onnx')
    # onnx.checker.check_model(optimized_model)
    print("ONNX export completed and verified")
    print("Model optimization completed")
    # print(optimized_model)
    # deployment = Deployment(quantized_model, 'llvm')
    # mod, params = deployment.deploy(inputs)
    # output, runtime = deployment.run_tvm_model(mod, params, 'input', inputs)
    # print("tvm output: ", output)   

if __name__ == '__main__':
    main()

    

        
