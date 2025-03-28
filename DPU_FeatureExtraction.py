import tvm
from tvm import dlight, relax, te, tir
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm.runtime import ShapeTuple
from tvm.contrib.download import download_testdata
from tvm.relay.quantize import quantize
from tvm.contrib.target import vitis_ai
from tvm.contrib import utils, graph_executor, cc
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai
import os 
from tvm import relay
from tqdm import tqdm
from datasets import load_dataset, Audio
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import onnx 
import onnxruntime
from tvm.relay.frontend import from_onnx
import pyxir.contrib.target.DPUCZDX8G


from transformers import Wav2Vec2Config, Wav2Vec2Processor, Wav2Vec2ForCTC

#Initialize:

MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
CONFIG = Wav2Vec2Config.from_pretrained(MODEL_NAME)
TVM_TARGET = 'llvm'
DPU_TARGET = 'DPUCZDX8G-kv260'

class VIVOSDataManager:
    """Handles VIVOS dataset loading and preprocessing."""
    
    def __init__(self, processor_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h", 
                 batch_size: int = 1) -> None:
        """Initialize VIVOS data manager.
        
        Args:
            processor_name: HuggingFace model ID for Wav2Vec2 processor
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
    
    def load_vivos(self, split: str = "train") -> Any:
        """Load VIVOS dataset.
        
        Args:
            split: Dataset split to use ('train', 'test', 'validation')
            
        Returns:
            VIVOS dataset
        """
        print(f"Loading VIVOS {split} dataset...")
        dataset = load_dataset("AILAB-VNUHCM/vivos", split=split)
        
        # Make sure audio is loaded with correct sampling rate
        if not hasattr(dataset.features, 'audio') or dataset.features.audio.sampling_rate is None:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        return dataset
    
    def prepare_batch_dataset(self, dataset: Any, max_samples: int = None) -> List[Dict[str, np.ndarray]]:
        """Prepare batched dataset for model input."""
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Preparing {len(dataset)} audio samples...")
        batched_inputs = []
        
        # First check the structure of the dataset
        first_item = dataset[0]
        print(f"Dataset sample structure: {list(first_item.keys())}")
        
        # Determine which field contains the transcription
        text_field = None
        for possible_field in ['text', 'sentence', 'transcription', 'transcript', 'label']:
            if possible_field in first_item:
                text_field = possible_field
                print(f"Found text field: '{text_field}'")
                break
        
        if not text_field:
            print("Warning: No text field found. Continuing without transcriptions.")
        
        for idx, item in enumerate(tqdm(dataset)):
            try:
                # Get audio data
                audio_data = item["audio"]["array"]
                sampling_rate = item["audio"]["sampling_rate"]
                
                # Process with Wav2Vec2 processor
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=sampling_rate, 
                    return_tensors="np"
                )
                
                # Create batch input with any available text
                batch_input = {"input": inputs.input_values}
                
                # Add text if available
                if text_field and text_field in item:
                    batch_input["text"] = item[text_field]
                
                # Add to batched inputs
                batched_inputs.append(batch_input)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
        
        print(f"Successfully prepared {len(batched_inputs)} audio batches")
        return batched_inputs
    
    def create_calibration_dataset(self, num_samples: int = 10) -> List[np.ndarray]:
        """Create calibration dataset from VIVOS.
        
        Args:
            num_samples: Number of samples to use for calibration
            
        Returns:
            List of calibration data samples
        """
        print("Creating calibration dataset from VIVOS...")
        dataset = self.load_vivos("train")
        batched_inputs = self.prepare_batch_dataset(dataset, num_samples)
        
        # Extract just the input values for calibration
        calibration_data = [batch["input"] for batch in batched_inputs]
        return calibration_data
    
    
class TVMCOmpiler:
    def __init__(self, model_path: str, tvm_target: str = "llvm", dpu_target: str = "DPUCZDX8G-kv260") -> None:
        """Initialize TVM compiler.
        
        Args:
            target: TVM target to compile for
            device: TVM device to compile for
        """
        self.target = tvm_target
        self.device = dpu_target
        self.model_path = model_path
        
        
    def load_onnx_model(self) -> Any:
        """Load ONNX model from file."""
        onnx_model = onnx.load(self.model_path)
        return onnx_model
    
    def convert_onnx_to_tvm(self, vitis_ai: bool=True) -> Any:
        """Convert ONNX model to TVM Relay."""
        onnx_model = self.load_onnx_model()
        input_name = onnx_model.graph.input[0].name
        print(f"Found input name in ONNX model: '{input_name}'")
    
    # Use the actual input name from the model
        shape_dict = {input_name: (1, 1, 86000)}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            
        if vitis_ai:
            mod = partition_for_vitis_ai(mod, params, dpu=DPU_TARGET)
        return mod, params
        
    def build_model(self, opt_level: int =3) -> Any:
        """Build TVM model."""
        print("Convert model to tvm format and adding vitis-ai partitioning")
        mod, params = self.convert_onnx_to_tvm(self.model_path)
        
        print("Building TVM model...")
        export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
        build_options = {
            'dpu': DPU_TARGET,
            'export_runtime_module': export_rt_mod_file
        }
        with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
            lib = relay.build(mod, TVM_TARGET, params=params)
        
        print("Finish build....")
        input_name = "input_values"
        module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
        dataloader = VIVOSDataManager()
        calibration_data = dataloader.create_calibration_dataset(num_samples=5)
        
        #Execute on-the-fly quantization
        for data in tqdm(enumerate(calibration_data)):
            module.set_input(input_name, data)
            module.run()
                
        temp = utils.tempdir()
        lib.export_library(temp.relpath("tvm_lib.so"))

        # Build and export lib for aarch64 target
        tvm_target = tvm.target.arm_cpu('ultra96')
        lib_kwargs = {
            'fcompile': cc.create_shared,
            'cc': "/usr/aarch64-linux-gnu/bin/ld"
        }

        build_options = {
            'load_runtime_module': export_rt_mod_file
        }
        with tvm.transform.PassContext(opt_level=1, config={'relay.ext.vitis_ai.options': build_options}):
            lib_edge = relay.build(mod, tvm_target, params=params)

        lib_edge.export_library('feature_extractor.so', **lib_kwargs)    
        

def __main__():
    model_path = "wav2vec2_feature_encoder.onnx"
    compiler = TVMCOmpiler(model_path)
    compiler.build_model()

if __name__ == "__main__":
    __main__()