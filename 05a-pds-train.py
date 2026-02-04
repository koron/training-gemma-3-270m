#!env python


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

############################################################################
# 定数

base_model = "google/gemma-3-270m-it"
data_file = "pds.csv"
checkpoint_dir = "checkpoints/pds"
learning_rate = 5e-5
train_epochs=5
train_batch_size=16

############################################################################
# データセットを準備

from datasets import load_dataset
dataset = load_dataset("./dataset", data_files=data_file, split="train")
def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["japanese"]},
          {"role": "assistant", "content": str(sample["pds"])}
      ]
  }
dataset = dataset.map(
        create_conversation,
        remove_columns=dataset.features,
        batched=False)
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

############################################################################
# モデルのロード

logger.info("model loading: start")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
logger.info("model loading: end")

############################################################################
# パイプライン構築

from transformers import pipeline

from random import randint

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"])-1)
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

############################################################################
# トレーニング

logger.info("training: start")

from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=train_epochs,          # number of training epochs
    per_device_train_batch_size=train_batch_size, # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # Template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

logger.info("training: end")
