class SemanticUncertaintyExtractor():
    def __init__(self, dataset, model_name):
        self.dataset = dataset
        self.model_name = model_name
    
    def __call__(self):
        prompt = self.prepare_prompt()
        responses = self.generate_responses(prompt)
        df = self.post_process_responses(responses)
        return df
    
    def prepare_prompt(self):
        pass
    
    def generate_responses(self, prompt):
        responses = []
        for i in range(len(self.dataset)):
            response = self.generate_single_response(prompt, self.model_name)
            responses.append(response)
        return responses
    
    def generate_single_response(self, prompt, model_name):
        pass
    
    def post_process_responses(self, responses):
        pass