
import torch
import argparse
import contexttimer

from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2

class Decoder:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def decode(self, t : torch.Tensor) -> str:
        # assert t.dim == 2, "t must be 2d tensor"
        return self.tokenizer.decode(t[0], skip_special_tokens=True)

DECODER : Decoder = None    

# my local models
MODELZOO = {
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    # https://huggingface.co/huggyllama/llama-13b
    "llama13b": None,
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    parser.add_argument('--approx_model_name', type=str, default=MODELZOO["llama1b"])
    parser.add_argument('--target_model_name', type=str, default=MODELZOO["llama7b"])
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed')
    args = parser.parse_args()
    return args


def generate(input_text, approx_model_name, target_model_name, num_tokens=40, random_seed = None, verbose = False, use_benchmark = True):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
    
    global DECODER
    DECODER = Decoder(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, trust_remote_code=True).to(torch_device)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, trust_remote_code=True).to(torch_device)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 10
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    TEST_TIME = 10
    if use_benchmark:
        with contexttimer.Timer() as t:
            for _ in range(TEST_TIME):
                output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
        print(f"\n[benchmark] large (target) model autoregressive_sampling 10 times, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])}")
        

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        with contexttimer.Timer() as t:
            for _ in range(TEST_TIME): 
                output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
        print(f"\n[benchmark] small (approx) model autoregressive_sampling 10 times, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")
    
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        with contexttimer.Timer() as t:
            for _ in range(TEST_TIME): 
                output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
        print(f"\n[benchmark] speculative_sampling 10 times, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(args.input, args.approx_model_name, args.target_model_name, random_seed = args.seed, verbose=args.verbose)