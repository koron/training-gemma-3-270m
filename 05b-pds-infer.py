#!env python

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("setup logger")

############################################################################
# 定数

base_model = "google/gemma-3-270m-it"
data_file = "pds.csv"
checkpoint_dir = "./checkpoints/pds/checkpoint-50"

model_id = checkpoint_dir
#model_id = base_model

############################################################################
# 学習済みモデルのロード

logger.info("model loading: import")
from transformers import AutoTokenizer, AutoModelForCausalLM
logger.info("model loading: start")
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
# 推論

from transformers import pipeline

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def test_batch(dataset, batch_size=16):
    # 1. プロンプトのリストを作成
    prompts = [
            pipe.tokenizer.apply_chat_template(
                sample["messages"][:1], 
                tokenize=False, 
                add_generation_prompt=True
                ) for sample in dataset
            ]

    # 2. まとめて推論実行
    # ※ pipeline にリストを渡すと結果がリスト（またはイテレータ）で返ってくる
    outputs = pipe(
            prompts, 
            batch_size=batch_size, 
            max_new_tokens=256, 
            disable_compile=True
            )

    # 3. 結果の抽出と照合
    results = []
    for sample, prompt, out in zip(dataset, prompts, outputs):
        query = sample['messages'][0]['content']
        want  = sample['messages'][1]['content']

        # 生成されたテキストからプロンプト部分を除去
        full_text = out[0]['generated_text']
        got = full_text[len(prompt):].strip()

        is_match = (want == got)

        # 1件ずつ表示（必要に応じて）
        print(f"{query}, {want}, {got}, {is_match}")

        results.append({
            "query": query,
            "want": want,
            "got": got,
            "is_match": is_match
            })

    return results

def count_results(label, results):
    count = 0
    match = 0
    for item in results:
        count += 1
        if item['is_match']:
            match += 1
    print(f"{label}: accuracy={match / count} ({match}/{count})")

# Test with an unseen dataset
results_train = test_batch(dataset['train'])
print('---')
results_test = test_batch(dataset['test'])
print('---')
count_results("train", results_train)
count_results("test", results_test)
