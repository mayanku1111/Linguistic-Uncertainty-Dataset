#gpt-5-mini
python -m llm_linguistic_confidence_study qa_model=gpt-5-mini confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-gpt-5-mini.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-mini confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-gpt-5-mini.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-mini confidence_extractor=verbal_numerical_confidence pre_runned_batch=vanilla-vnc-gpt-5-mini.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-mini confidence_extractor=semantic_uncertainty pre_runned_batch=vanilla-su-gpt-5-mini.yaml

#gpt-5-nano
python -m llm_linguistic_confidence_study qa_model=gpt-5-nano confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-gpt-5-nano.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-nano confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-gpt-5-nano.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-nano confidence_extractor=verbal_numerical_confidence pre_runned_batch=vanilla-vnc-gpt-5-nano.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5-nano confidence_extractor=semantic_uncertainty pre_runned_batch=vanilla-su-gpt-5-nano.yaml

#gpt-5
python -m llm_linguistic_confidence_study qa_model=gpt-5 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-gpt-5.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-gpt-5.yaml
python -m llm_linguistic_confidence_study qa_model=gpt-5 confidence_extractor=verbal_numerical_confidence pre_runned_batch=vanilla-vnc-gpt-5.yaml

#Llama-4-Maverick-17B-128E-Instruct-FP8
python -m llm_linguistic_confidence_study qa_model=Llama-4-Maverick-17B-128E-Instruct-FP8 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-Llama-4-Maverick-17B-128E-Instruct-FP8.yaml
python -m llm_linguistic_confidence_study qa_model=Llama-4-Maverick-17B-128E-Instruct-FP8 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-Llama-4-Maverick-17B-128E-Instruct-FP8.yaml
python -m llm_linguistic_confidence_study qa_model=Llama-4-Maverick-17B-128E-Instruct-FP8 confidence_extractor=verbal_numerical_confidence pre_runned_batch=vanilla-vnc-Llama-4-Maverick-17B-128E-Instruct-FP8.yaml
python -m llm_linguistic_confidence_study qa_model=Llama-4-Maverick-17B-128E-Instruct-FP8 confidence_extractor=semantic_uncertainty pre_runned_batch=vanilla-su-Llama-4-Maverick-17B-128E-Instruct-FP8.yaml

#claude-sonnet-4-20250514
python -m llm_linguistic_confidence_study qa_model=claude-sonnet-4-20250514 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-claude-sonnet-4-20250514.yaml
python -m llm_linguistic_confidence_study qa_model=claude-sonnet-4-20250514 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-claude-sonnet-4-20250514.yaml

#claude-3-5-haiku-20241022
python -m llm_linguistic_confidence_study qa_model=claude-3-5-haiku-20241022 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-claude-3-5-haiku-20241022.yaml
python -m llm_linguistic_confidence_study qa_model=claude-3-5-haiku-20241022 confidence_extractor=linguistic_confidence pre_runned_batch=vanilla-uncertainty-claude-3-5-haiku-20241022.yaml
python -m llm_linguistic_confidence_study qa_model=claude-3-5-haiku-20241022 confidence_extractor=verbal_numerical_confidence pre_runned_batch=vanilla-vnc-claude-3-5-haiku-20241022.yaml
python -m llm_linguistic_confidence_study qa_model=claude-3-5-haiku-20241022 confidence_extractor=semantic_uncertainty pre_runned_batch=vanilla-su-claude-3-5-haiku-20241022.yaml
