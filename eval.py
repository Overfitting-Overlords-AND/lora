#%%
from tokens import tokenizer
import transformers as t
import torch
import peft
import config
#%%
tokenizer.pad_token_id = 0
base_model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
#%%
with open('./prompt.txt', 'r') as file:
  TEMPLATE = file.read()
QUESTION = "How many teachers are earning less than £40000"
CONTEXT = "CREATE TABLE teacher (salary INTEGER)"
prompt = TEMPLATE.format(question=QUESTION,context=CONTEXT)
#%%
if not config.BASE_MODEL:
  model = peft.PeftModel.from_pretrained(base_model, "./output/checkpoint-800")
  pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
else: 
  pipe = t.pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=500)

#%%
print("pipe(prompt)", pipe(prompt))
