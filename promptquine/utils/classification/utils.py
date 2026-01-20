from transformers import AutoConfig

def is_masked_language_model(model_name):
    """Check if the model is a Masked Language Model."""
    config = AutoConfig.from_pretrained(model_name)
    
    model_type = config.model_type.lower()
    
    masked_lm_types = {
        'bert', 'roberta', 'albert', 'electra', 
        'distilbert', 'xlm-roberta', 'ernie'
    }
    
    return model_type in masked_lm_types