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

## さまざまな数値表現のアラビア数字化

-   データセット [dataset/janum.csv](./dataset/janum.csv)
-   [./02a-janum-train.py](./02a-janum-train.py) 学習用
-   [./02b-janum-infer.py](./02b-janum-infer.py) 推論用(テストデータセットのみ)
-   [./02c-janum-infer-all.py](./02c-janum-infer-all.py) 推論用(全データ対象)

|  japanese  |   arabic   | inference | match  |
|------------|------------|-----------|--------|
| 五分五分   |  (0.5)     |  (5)      |  False |
| 半分       |  (0.5)     |  (2)      |  False |
| 四半       |  (0.25)    |  (25)     |  False |
| 八分目     |  (0.8)     |  (8)      |  False |
| 単身       |  (1)       |  (1)      |  True  |
| コンビ・対 |  (2)       |  (3)      |  False |
| 唯一無二   |  (1)       |  (1)      |  True  |
| 四六時中   |  (24)      |  (6)      |  False |
| 三羽ガラス |  (3)       |  (3)      |  True  |
| 八百万     |  (8000000) |  (80000)  |  False |

-   結果 (accuracy = 3/10 = 0.3) は良いとは言えない
    -   そもそも学習データの内容が良くない
        - 並びのせいでtrainとtestの組み分けが恣意的
        - 件数が少ない
        - 組み合わせが少ない
    -   浮動小数点に弱い
    -   言葉の一部や成句に引き摺られていそう
        -   半分 → 1/2 の `2`
        -   四半 → 四半世紀 → `25`
        -   四六時中 → 最初の `4`
-   カッコで囲ったのは複数の都合による
    -   カッコ無しだと列が数値(float)として扱われ、文字列表現に癖が出て (`1 → 1.0`, `10 → 10.0`) その表現が学習に対するノイズになる
    -   開始と終了を個別にマークしたほうが結果が良かった
    -   このルールは完璧に学習したようだ

参考: [全データに対する推論](results/02c-janum-all.csv)

## 数値表現のアラビア数字可 + データシャッフル

データセットの並びとtrainとtesutの組み分けが恣意的で、学習結果に悪い影響がでていないかを検証する。
そのためにデータセットをランダムに入れ替えたのちに、同じように学習推論をしてみる。

結果: accuracy = 7/10 = 0.7

-   データセット [dataset/janum+shuf.csv](./dataset/janum+shuf.csv)
-   [./03a-janum+shuf-train.py](./03a-janum+shuf-train.py) 学習用
-   [./03b-janum+shuf-infer-all.py](./03b-janum+shuf-infer-all.py) 推論用(全データ対象)

|  japanese  | arabic | inference | match  |
|------------|--------|-----------|--------|
| 廿         |  (20)  |  (20)     |  True  |
| コンビ・対 |  (2)   |  (2)      |  True  |
| 十         |  (10)  |  (10)     |  True  |
| 十四日     |  (14)  |  (14)     |  True  |
| 参         |  (3)   |  (100000) |  False |
| 単身       |  (1)   |  (1)      |  True  |
| 二十日     |  (20)  |  (2)      |  False |
| 三日       |  (3)   |  (3)      |  True  |
| 一割       |  (0.1) |  (0.25)   |  False |
| 七つ       |  (7)   |  (7)      |  True  |

-   結果 accuracy = 0.7 は良くなった
    -   経験的に納得のいく割合
-   得手・不得手はありそう
    -   1文字目の数字は読みやすい
    -   2文字目以降が重要だと間違えやすい: 例: `二十日`
    -   少数の扱いは苦手そう

参考: [全データに対する推論](results/03b-janum+shuf.csv)
