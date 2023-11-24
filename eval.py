#%%
from tokens import tokenizer
import transformers as t
import torch
import peft
import config
#%%
tokenizer.pad_token_id = 0
base_model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=False, torch_dtype=torch.float)
# base_model.resize_token_embeddings(len(tokenizer))
#%%
with open('./prompt.txt', 'r') as file:
  TEMPLATE = file.read()
QUESTION = "How many teachers are earning less than £40000"
CONTEXT = "CREATE TABLE teacher (salary INTEGER)"
# SELECT COUNT(*) FROM list AS T1 JOIN teachers AS T2 ON T1.classroom = T2.classroom WHERE T1.firstname = “CHRISSY” AND T1.lastname = “NABOZNY”

prompt = TEMPLATE.format(question=QUESTION,context=CONTEXT)
#%%
if not config.BASE_MODEL:
  model = peft.PeftModel.from_pretrained(base_model, "./output/checkpoint-2200")
  pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)
else: 
  pipe = t.pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=100)

#%%
print("pipe(prompt)", pipe(prompt))
