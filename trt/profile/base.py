from abc import ABC, abstractmethod
from lerobot.policies.factory import make_pre_post_processors

from trt.data import (
    load_test_data, 
    frame_from_test_data
)

class VLAProfile:
    """One object owns everything HuggingFace/LeRobot needs before export runs."""
    def __init__(self, device, model_id):
        self.device = device
        self.model_id = model_id

        self._init_policy()
        self._init_models()
        self._init_processors()
        self._init_tokenizers()
            
    # private
    @abstractmethod
    def _init_policy(self):
        ...

    @abstractmethod
    def _init_models(self):
        ...
        
    @abstractmethod
    def _init_processors(self):
        self.pre_processor, self.post_processor = make_pre_post_processors(
            self.config,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
    
    @abstractmethod
    def _prepare_model_inputs(self, data, fill_missing=True) -> dict:
        frame = frame_from_test_data(data, self.policy, fill_missing=fill_missing)
        model_inputs = self.pre_processor(frame)
        return model_inputs
    
    @abstractmethod
    def _init_tokenizers(self):
        self.text_tok = getattr(self.pre_processor, "tokenizer", None)
    
    @abstractmethod
    def pre_process(self):
        ...