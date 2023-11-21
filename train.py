import transformers
import model
import data
import os
import wandb


is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
m = model.get_model()

dataset = data.load_dataset("b-mc2/sql-create-context")
train_dataset, eval_dataset = dataset["train"].train_test_split(test_size=0.1).values()
train_dataset, eval_dataset = data.DatasetReader(train_dataset), data.DatasetReader(eval_dataset)

collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

wandb.init(project="lora_text_to_sql")

# import torch
# import peft
# adapters_weights = torch.load("./output/checkpoint-800/adapter_model.bin")
# peft.set_peft_model_state_dict(m, adapters_weights)


trainer = transformers.Trainer(
  model=m,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  data_collator=collator,
  args=transformers.TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    output_dir="./output",
    save_total_limit=3,
    ddp_find_unused_parameters=False if is_ddp else None,
    report_to="wandb",
  ),
)


m.config.use_cache = False
trainer.train()
m.save_pretrained("./weights")
