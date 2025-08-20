from .linguistic_confidence import LinguisticConfidenceExtractor
from .pture import PTrueConfidenceExtractor
from .semantic_uncertainty import SemanticUncertaintyExtractor
from .verbal_numerical_confidence import VerbalNumericalConfidenceExtractor

from omegaconf import DictConfig

class ConfidenceExtractor:
    def __init__(self, confidence_extraction_method_cfg: DictConfig, dataset_cfg: DictConfig, model_cfg: DictConfig):
        self.confidence_extractor = self.get_confidence_extractor(confidence_extraction_method_cfg.name)
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
    
    def __call__(self, dataset_df):
        df = self.confidence_extractor(dataset_df, self.model_cfg)
        return df
    
    def prepare_model(self, model_cfg):
        pass
    
    def get_confidence_extractor(self, confidence_extraction_method_name):
        if confidence_extraction_method_name == "linguistic_confidence": 
            return LinguisticConfidenceExtractor(self.dataset_cfg, self.model_cfg)
        elif confidence_extraction_method_name == "pture":
            return PTrueConfidenceExtractor(self.dataset_cfg, self.model_cfg)
        elif confidence_extraction_method_name == "semantic_uncertainty":
            return SemanticUncertaintyExtractor(self.dataset_cfg, self.model_cfg)
        elif confidence_extraction_method_name == "verbal_numerical_confidence":
            return VerbalNumericalConfidenceExtractor(self.dataset_cfg, self.model_cfg)
        else:
            raise ValueError(f"Invalid confidence extraction method: {confidence_extraction_method_name}")