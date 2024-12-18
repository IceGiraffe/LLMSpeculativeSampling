from sampling.kvcache_model import KVCacheModel
import torch
# model_name = "bigscience/bloomz-7b1"
model_name = "/root/autodl-tmp/huggingface/hub/bloomz-7b1"
# argparse the model_name and input_text
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--input_text", type=str, default="Hello world")
    return parser.parse_args()

args = parse_args()
model_name = args.model_name
input_text = args.input_text

from transformers import AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda")
with torch.no_grad():

    model = BloomForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.float16)
    model = model.to("cuda")
    kv_model = KVCacheModel(model)
    output = kv_model.generate(input_ids, 100)
    generated_text = tokenizer.decode(output[0])
    print(generated_text)