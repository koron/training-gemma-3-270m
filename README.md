# Gemma 3 270M で独自の学習をやってみる

ベースとするモデルは [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)

お題としては以下のような簡単なモノ

*   様々な数値表現を正規化 (半角の 0-9 で表現) する
*   レシピの材料の量を倍化する (大さじ 1/2 x3 → 3/2)

学習方法はフルモデルトレーニングとLoRAが考えられる。

もう一度 <https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune?hl=ja> を実践するところから初めても良いかも。

## 環境のセットアップ

```
$ python -m venv venv

(ここは環境によって対応方法が異なる)
$ source ./venv/Scripts/activate

(例によって pip の更新)
$ python -m pip install -U pip

(ここも環境で変わる。詳しくは https://pytorch.org/get-started/locally/)
$ pip3 install torch --index-url https://download.pytorch.org/whl/cu128

$ pip3 install tensorboard

$ pip3 install transformers datasets accelerate evaluate trl protobuf sentencepiece

$ pip3 install matplotlib
```

2回目以降は以下だけで良い。

```
$ source ./venv/Scripts/activate
```

## 火星人の例: 再実施

以前やっていたことを思い出すために再実施。
フルモデルのトレーニングになるはず。

データ: [bebechien/MobileGameNPC](https://huggingface.co/datasets/bebechien/MobileGameNPC)

- [./01-martian-train.py](./01-martian-train.py) 学習用
- [./01-martian-infer.py](./01-martian-infer.py) 推論用

    <details>
    <summary>出力</summary>

    ```
    Question:
    Do you know any jokes?
    Original Answer:
    A joke? k'tak Yez. A Terran, a Glarzon, and a pile of nutrient-pazte walk into a bar... Narg, I forget da rezt. Da punch-line waz zarcaztic.
    Generated Answer:
    Yez! Yez! Diz Diz... k'tak... Diz iz... 'em... fuzzy. Why you burn them with acid?
    --------------------------------------------------------------------------------
    Question:
    (Stands idle for too long)
    Original Answer:
    You'z broken, Terran? Or iz diz... 'meditation'? You look like you're trying to lay an egg.
    Generated Answer:
    Diz? Diz iz da Iez.
    --------------------------------------------------------------------------------
    Question:
    What do you think of my outfit?
    Original Answer:
    Iz very... pointy. Are you expecting to be attacked by zky-eelz? On Marz, dat would be zenzible.
    Generated Answer:
    My look. It iz... k'tak... minimalist. Why you burn it with fire first?
    --------------------------------------------------------------------------------
    Question:
    It's raining.
    Original Answer:
    Gah! Da zky iz leaking again! Zorp will be in da zhelter until it ztopz being zo... wet. Diz iz no good for my jointz.
    Generated Answer:
    Diz? Diz iz da Outpozt. May your zpare Jellypod live long... wet.
    --------------------------------------------------------------------------------
    Question:
    I brought you a gift.
    Original Answer:
    A gift? For Zorp? k'tak It iz... a small rock. Very... rock-like. Zorp will put it with da other rockz. Thank you for da thought, Terran.
    Generated Answer:
    A genuine Martian grav-exo-fruit. Very... zodium. Why you bring it to me?
    --------------------------------------------------------------------------------
    ```

    </details>

以下のようなエラーが出たか、動いているように見える。

> The tokenizer you are loading from './checkpoints/MyGemmaNPC/checkpoint-25' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.

