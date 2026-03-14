from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=config.DEVICE,
    low_cpu_mem_usage=True,
)

class Model:
    tokenizer = tokenizer
    model     = model








