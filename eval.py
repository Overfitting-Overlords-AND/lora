#%%
import transformers as t
import torch
import peft
import time
#%%
tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
tokenizer.pad_token_id = 0
#%%
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
model = peft.get_peft_model(model, config)
peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-400"))
#%%
with open('./prompt.txt', 'r') as file:
  TEMPLATE = file.read()
QUESTION = "How many teachers are earning less than £40000"
CONTEXT = "CREATE TABLE teacher (salary INTEGER)"
prompt = TEMPLATE.format(question=QUESTION,context=CONTEXT)
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))
