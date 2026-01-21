#!env python

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

############################################################################
# 定数

checkpoint_dir = "./checkpoints/janum/checkpoint-50"

############################################################################
# 学習済みモデルのロード

logger.info("model loading: import")
from transformers import AutoTokenizer, AutoModelForCausalLM
logger.info("model loading: start")
model_id = checkpoint_dir
# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
logger.info("model loading: end")

############################################################################
# データセットを準備

from datasets import load_dataset
dataset = load_dataset("./dataset", data_files="janum.csv", split="train")
def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["japanese"]},
          {"role": "assistant", "content": str(sample["arabic"])}
      ]
  }
dataset = dataset.map(
        create_conversation,
        remove_columns=dataset.features,
        batched=False)
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

############################################################################
# 推論

from transformers import pipeline

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def test(test_sample):
  # Convert as test example into a prompt with the Gemma template
  prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
  outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

  # Extract the user query and original answer
  print(f"Question:\n{test_sample['messages'][0]['content']}")
  print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
  print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
  print("-"*80)

# Test with an unseen dataset
for item in dataset['test']:
  test(item)
