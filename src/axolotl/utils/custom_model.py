import shutil
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from ..custom_model.configuration_phi3 import Phi3WithExtraModuleConfig
from ..custom_model.modeling_phi3 import Phi3WithExtraModuleForCausalLM

def copy_and_replace(source_file, destination_directory):
    filename = os.path.basename(source_file)
    destination_path = os.path.join(destination_directory, filename)
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy(source_file, destination_path)

def register_to_auto_class():
    AutoConfig.register("phi3_custom", Phi3WithExtraModuleConfig)
    AutoModelForCausalLM.register(Phi3WithExtraModuleConfig, Phi3WithExtraModuleForCausalLM)

def get_custom_model(model_url):
    config = None

    if "extra" in model_url:
        if "Phi-3" in model_url:
            if "lora" in model_url:
                copy_and_replace("../custom_model/configuration_phi3.py", model_url)
                copy_and_replace("../custom_model/modeling_phi3.py", model_url)
                register_to_auto_class()
                config = AutoConfig.from_pretrained(model_url)
    elif "prune" in model_url:
        split_list = model_url.split("/")[-1].split("-")
        prune_start_index, prune_end_index = int(split_list[-2]), int(split_list[-1])
        if "Phi-3" in model_url:
            copy_and_replace("../custom_model/config.json", model_url)
            copy_and_replace("../custom_model/configuration_phi3.py", model_url)
            copy_and_replace("../custom_model/modeling_phi3.py", model_url)
            register_to_auto_class()
            config = AutoConfig.from_pretrained(model_url)
            config.num_hidden_layers = 32 - (prune_end_index - prune_start_index + 1)
    else:
        config = AutoConfig.from_pretrained(model_url)
        
    
    model = AutoModelForCausalLM.from_pretrained(model_url, config=config, torch_dtype="auto", device_map="cuda")

    return model