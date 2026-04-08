from transformers import AutoTokenizer, AutoModelForCausalLM

model = "meta-llama/Llama-3.2-1B-Instruct"

tok = AutoTokenizer.from_pretrained(model)
mdl = AutoModelForCausalLM.from_pretrained(model)
print("OK")