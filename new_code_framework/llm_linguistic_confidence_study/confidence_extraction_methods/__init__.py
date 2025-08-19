from .linguistic_confidence import LinguisticConfidenceExtractor
from .pture import PTrueConfidenceExtractor
from .semantic_uncertainty import SemanticUncertaintyExtractor
from .verbal_numerical_confidence import VerbalNumericalConfidenceExtractor

class ConfidenceExtractor:
    def __init__(self, confidence_extraction_method, dataset, model_name):
        self.confidence_extractor = self.get_confidence_extractor(confidence_extraction_method)
        self.dataset = dataset
        self.model_name = model_name
    
    def __call__(self):
        df = self.confidence_extractor(self.dataset, self.model_name)
        return df
    
    def get_confidence_extractor(self, confidence_extraction_method):
        if confidence_extraction_method == "linguistic_confidence": 
            return LinguisticConfidenceExtractor(self.dataset, self.model_name)
        elif confidence_extraction_method == "pture":
            return PTrueConfidenceExtractor(self.dataset, self.model_name)
        elif confidence_extraction_method == "semantic_uncertainty":
            return SemanticUncertaintyExtractor(self.dataset, self.model_name)
        elif confidence_extraction_method == "verbal_numerical_confidence":
            return VerbalNumericalConfidenceExtractor(self.dataset, self.model_name)
        else:
            raise ValueError(f"Invalid confidence extraction method: {confidence_extraction_method}")