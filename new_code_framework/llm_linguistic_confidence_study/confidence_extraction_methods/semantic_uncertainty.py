class SemanticUncertaintyExtractor():
    def __init__(self, dataset, model_name):
        self.dataset = dataset
        self.model_name = model_name
    
    def __call__(self, dataset, model_name):
        prompt = self.prepare_prompt()
        responses = self.generate_responses(prompt, dataset, model_name)
        df = self.post_process_responses(responses)
        return df    # return a dataframe with the following columns: question, gold_answer, reponse1, reponse2, reponse3, ..., confidence, accuracy
    
    def prepare_prompt(self):
        pass
    
    def generate_responses(self, prompt, dataset, model_name):
        responses = []
        for i in range(len(dataset)):
            response = self.generate_single_response(prompt, dataset, model_name)
            responses.append(response)
        return responses
    
    def generate_single_response(self, prompt, model_name):
        pass
    
    def post_process_responses(self, responses):
        
        pass
    
    def calculate_confidence(self, responses, gold_answer):
        pass
    
    def calculate_accuracy(self, responses, gold_answer):
        pass
    