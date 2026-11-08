# japanese-bm25-sparse-embeddings

QdrantのBM25（FastEmbedの`Qdrant/bm25`）を日本語で使うための検証コードです。

SudachiPyで分かち書きしてからFastEmbedに渡す`JapaneseBM25SparseEmbeddings`と、「分かち書きなし」と比較するノートブックを含みます。

- 関連記事（Zenn）: 公開後にURLを追記します

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuya-fukuoka/japanese-bm25-sparse-embeddings/blob/main/japanese_bm25_colab_v2.ipynb)

## 背景

FastEmbedの`Qdrant/bm25`は、**空白などで区切られたトークン**を前提にしています。内部のトークナイザは`\w`以外の文字で分割しますが、Pythonの正規表現では日本語の文字も`\w`に含まれます。そのため、日本語の文をそのまま渡すと、空白や句読点で挟まれたかたまりが1トークンになり、日本語の語が照合されません。

このリポジトリでは、SudachiPyで名詞を取り出してから渡す方法を試し、分かち書きなしの場合と比較しています。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `japanese_bm25_colab_v2.ipynb` | 5方式×7クエリの比較検証。実行結果つき（Google Colabで実行） |
| `LangChain_Qdrant_BM25.ipynb` | 初期の検証ノートブック。記事の改訂前のもので、`.lower()`による小文字化を含みません |
| `LICENSE` | MIT License |

## 比較した5方式

| 方式 | 前処理 | avg_len |
| --- | --- | --- |
| A | なし（そのまま`Qdrant/bm25`へ） | 既定値（256） |
| A2 | なし | 実データの平均 |
| B | Sudachiの名詞を`normalized_form()`でhex化 | 実データの平均 |
| C | Bに`.lower()`を追加 | 実データの平均 |
| C256 | Cと同じ | 既定値（256） |

## 検証結果の要約

6件の文書と7つのクエリ（英字を含む3つ、日本語の語だけの4つ）での結果です。

| 方式 | 正解が1位 | 検索結果が0件 |
| --- | --- | --- |
| A | 2 / 7 | 5 / 7 |
| A2 | 2 / 7 | 5 / 7 |
| B | 7 / 7 | 0 / 7 |
| C | 7 / 7 | 0 / 7 |
| C256 | 7 / 7 | 0 / 7 |

- 方式Aで正解が1位になった2件も、一致したのは英字1トークン（`aws`、`c`）だけでした
- 方式Aは、`token_max_length`（既定40文字）を超えるかたまりを捨てます。文書005は、Sparse Vectorの要素が1つだけになりました
- Sudachiの`normalized_form()`は、語によって大文字のまま残すか小文字にするかが変わります。`.lower()`がない方式Bでは、クエリの`Python`が文書の`python`に一致せず、Pythonを含む文書が検索結果から漏れました
- `avg_len`を変えても（AとA2、CとC256）、順位は変わりませんでした。スコアの値は変わります

詳しい数値と考察は、関連記事とノートブックの実行結果を参照してください。

## 使い方

### Google Colab

上のバッジから開き、上から順に実行してください。

### ローカル

```bash
pip install "langchain-qdrant==1.1.0" "qdrant-client==1.19.0" "fastembed==0.8.0" "SudachiPy==0.6.11" sudachidict_core pandas
```

ノートブックの`Qdrant/bm25`のダウンロードには、Hugging Faceへの接続が必要です。

### 最小のコード例

ノートブックの実装を抜粋したものです。

```python
from langchain_core.documents import Document
from langchain_qdrant import (
    FastEmbedSparse, QdrantVectorStore, RetrievalMode, SparseEmbeddings,
)
from qdrant_client import QdrantClient, models
from sudachipy import Dictionary, SplitMode


class SudachiBM25Tokenizer:
    """SplitMode.Cで分割し、名詞のnormalized_form()を小文字で返す"""

    def __init__(self, split_mode=SplitMode.C, target_pos=None, lower=True):
        self.tokenizer = Dictionary().create()
        self.split_mode = split_mode
        self.target_pos = target_pos or {"名詞"}
        self.lower = lower

    def __call__(self, text):
        tokens = []
        for m in self.tokenizer.tokenize(text, self.split_mode):
            if m.part_of_speech()[0] not in self.target_pos:
                continue
            token = m.normalized_form().strip()
            if self.lower:
                token = token.lower()
            if token:
                tokens.append(token)
        return tokens


class JapaneseBM25SparseEmbeddings(SparseEmbeddings):
    """SudachiPy + FastEmbedSparse(Qdrant/bm25)。トークンはUTF-8のhex文字列にして渡す"""

    def __init__(self, tokenize, avg_len, *, k=1.5, b=0.75):
        self.tokenize = tokenize
        self.bm25 = FastEmbedSparse(
            model_name="Qdrant/bm25",
            disable_stemmer=True,
            k=k, b=b,
            avg_len=avg_len,
            token_max_length=512,   # hex化で文字数が増える
        )

    @staticmethod
    def _encode_token(token):
        return "t" + token.encode("utf-8").hex()

    def preprocess(self, text):
        return " ".join(self._encode_token(t) for t in self.tokenize(text))

    def embed_documents(self, texts):
        return self.bm25.embed_documents([self.preprocess(t) for t in texts])

    def embed_query(self, text):
        return self.bm25.embed_query(self.preprocess(text))


documents = [
    Document(page_content="Pythonを使用し、AWS Lambdaを利用したAPIの開発経験があります。", metadata={"id": "doc-1"}),
    Document(page_content="COBOLを使用し、銀行の勘定系システムの保守と開発経験があります。", metadata={"id": "doc-2"}),
]

tokenize = SudachiBM25Tokenizer()
avg_len = sum(len(tokenize(d.page_content)) for d in documents) / len(documents)  # 実データから計算する

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="demo",
    # FastEmbedのBM25はTF部分だけを持つ。IDFはQdrant側で計算するため、Modifier.IDFが必須
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

store = QdrantVectorStore(
    client=client,
    collection_name="demo",
    sparse_embedding=JapaneseBM25SparseEmbeddings(tokenize, avg_len),
    sparse_vector_name="bm25",
    retrieval_mode=RetrievalMode.SPARSE,
)
store.add_documents(documents)

for doc, score in store.similarity_search_with_score("銀行の勘定系システムの保守", k=2):
    print(doc.metadata["id"], round(score, 3))
```

tokenizerを変更した場合は、コレクションを作り直して登録し直してください。文書は登録時に一度だけトークン化されます。

## 動作確認した環境

| パッケージ | バージョン |
| --- | --- |
| langchain-qdrant | 1.1.0 |
| qdrant-client | 1.19.0 |
| fastembed | 0.8.0 |
| SudachiPy | 0.6.11 |
| SudachiDict-core | 20260723 |

実行環境はGoogle Colabです。他のバージョンでは確認していません。特に、方式Aの挙動はfastembed 0.8.0のソースに基づきます。

## 制約・注意

- 文書は6件、クエリは7件、正解は各クエリ1件です。順位と一致トークン数の確認までで、再現率や適合率は評価していません
- 文書とクエリは、検証用に作成した短い架空の経歴文です。実データでは結果が変わる可能性があります
- 名詞だけを使うため、動詞や形容詞は照合されません
- `C++`は`c`になります。Sudachiが記号の`++`を落とすためです
- BM25は語の出現だけを見るので、否定（「経験なし」と「経験あり」）は区別できません

## 今後の検証

次の3点は、別の記事で検証する予定です。

1. 方式Aで`token_max_length`を伸ばした場合の挙動
2. 文書数・文書長のばらつきを増やした場合の`avg_len`の影響
3. 数万件規模での動作

## ライセンス

[MIT License](LICENSE)

依存ライブラリ（LangChain、Qdrant、FastEmbed、SudachiPy、SudachiDict）と、`Qdrant/bm25`モデルのライセンスは、それぞれのプロジェクトを参照してください。
