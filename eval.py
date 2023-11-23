#%%
from tokens import tokenizer
import transformers as t
import torch
import peft
import time
#%%
tokenizer.pad_token_id = 0
base_model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
#%%
# config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
# model = peft.get_peft_model(model, config)
# peft.set_peft_model_state_dict(model, model.from_pretrained("checkpoint-800"))
config = peft.PeftConfig.from_pretrained("./output/checkpoint-800")
model = peft.PeftModel.from_pretrained(base_model, "./output/checkpoint-800")
#%%
with open('./prompt.txt', 'r') as file:
  TEMPLATE = file.read()
QUESTION = "How many teachers are earning less than £40000"
CONTEXT = "CREATE TABLE teacher (salary INTEGER)"
prompt = TEMPLATE.format(question=QUESTION,context=CONTEXT)
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))
