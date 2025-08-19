from .verbal_confidence_extractor import VerbalConfidenceExtractor
from .numerical_confidence_extractor import NumericalConfidenceExtractor


class ConfidenceExtractor:
    def __init__(self, confidence_extraction_method, qa_dataset):
        self.confidence_extraction_method = confidence_extraction_method
        self.extractor = self.get_extractor()
        self.qa_dataset = qa_dataset
        

    def get_extractor(self):
        if self.confidence_extraction_method == "verbal":
            return VerbalConfidenceExtractor()
        elif self.confidence_extraction_method == "numerical":
            return NumericalConfidenceExtractor()
        else:
            raise ValueError(f"Invalid confidence extraction method: {self.confidence_extraction_method}")
        
    def generate_responses(self):
        pass
        
    def evaluate_responses(self):
        pass
        
    def save_results(self):
        pass
        
    