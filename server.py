import fastapi
from tokens import tokenizer
import transformers as t
import torch
import peft

app = fastapi.FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")
  app.state.tokenizer = tokenizer
  app.state.tokenizer.pad_token_id = 0
  app.state.base_model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=False, torch_dtype=torch.float)
  app.state.model = peft.PeftModel.from_pretrained(app.state.base_model, "./output/checkpoint-800")
  app.state.pipe = t.pipeline(task="text-generation", model=app.state.model, tokenizer=app.state.tokenizer, max_length=500)
  with open('./prompt.txt', 'r') as file:
    app.state.template = file.read()

@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")

@app.get("/")
def on_root():
  return { "message": "Hello App" }

@app.post("/from_text_to_sql")
async def text_to_sql(request: fastapi.Request):
  payload = await request.json()
  prompt = app.state.template.format(question=payload["txt"],context=payload["ctx"])
  answer = app.state.pipe(prompt)
  print("Input question: ", payload["txt"])
  print("Input context: ", payload["ctx"])
  return answer




