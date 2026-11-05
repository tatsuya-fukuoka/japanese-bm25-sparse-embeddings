# japanese-bm25-sparse-embeddings

Qdrant の BM25（FastEmbed の `Qdrant/bm25`）を、日本語のテキストで使うためのサンプルです。
SudachiPy で分かち書きし、LangChain の `QdrantVectorStore` からキーワード検索（Sparse 検索）を行います。

解説記事: <!-- Zenn記事のURLを記載 -->

## 概要

FastEmbed の `Qdrant/bm25` は、空白区切りの英語を前提にしています。日本語の文は単語が空白で区切られていないため、そのまま渡すと単語に分けられません。このリポジトリでは、次の流れで日本語に対応します。

```
日本語テキスト
  ↓ SudachiPy（SplitMode.C、名詞のみ、normalized_form）
トークン列
  ↓ UTF-8 の hex 文字列に変換
FastEmbedSparse（Qdrant/bm25、stemmer なし）
  ↓
SparseVector
  ↓
Qdrant（Modifier.IDF の Sparse Vector）
```

## 内容

| ファイル | 説明 |
| --- | --- |
| `LangChain_Qdrant_BM25.ipynb` | 実装と動作確認のノートブック |

ノートブックでは、次のことを順に行います。

1. SudachiPy の tokenizer を作り、分かち書きの結果を確認する
2. 登録する文書から `avg_len`（平均トークン数）を計算する
3. LangChain の `SparseEmbeddings` を継承した `JapaneseBM25SparseEmbeddings` を実装する
4. Qdrant（インメモリ）に、`Modifier.IDF` を指定した Sparse Vector のコレクションを作る
5. エンジニアの経歴文 6 件を登録し、BM25 で検索する

## 動作確認した環境

| パッケージ | バージョン |
| --- | --- |
| langchain-qdrant | 1.1.0 |
| qdrant-client | 1.19.0 |
| fastembed | 0.8.0 |
| SudachiPy | 0.6.11 |
| SudachiDict-core | 20260723 |

## インストール

```bash
pip install -U langchain-qdrant qdrant-client fastembed sudachipy sudachidict_core
```

## 使い方

ノートブックの中心になる部分だけを抜き出すと、次のとおりです。

```python
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode

# 1. 文書
documents = [
    Document(
        page_content="Pythonを3年間使用し、AWS Lambdaを利用したサーバーレスアプリケーションの設計・開発経験があります。",
        metadata={"id": "engineer-001"},
    ),
    # ...
]

# 2. tokenizer と avg_len（実データの平均トークン数）
tokenize = SudachiBM25Tokenizer()
avg_len = sum(len(tokenize(d.page_content)) for d in documents) / len(documents)

# 3. Sparse Embeddings
sparse_embeddings = JapaneseBM25SparseEmbeddings(
    tokenize=tokenize, avg_len=avg_len, k=1.5, b=0.75
)

# 4. コレクション（Modifier.IDF が必須）
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="engineer_bm25_demo",
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)

# 5. 登録と検索
vector_store = QdrantVectorStore(
    client=client,
    collection_name="engineer_bm25_demo",
    sparse_embedding=sparse_embeddings,
    sparse_vector_name="bm25",
    retrieval_mode=RetrievalMode.SPARSE,
)
vector_store.add_documents(documents)

results = vector_store.similarity_search_with_score(
    "Python AWS Lambdaを利用した開発経験", k=3
)
```

`SudachiBM25Tokenizer` と `JapaneseBM25SparseEmbeddings` の実装は、ノートブックを参照してください。

## 設計上のポイント

- **名詞のみを使います。** BM25 のキーワード検索なので、まず名詞中心で始めています。`SudachiBM25Tokenizer` の `target_pos` で変更できます。
- **トークンを hex 文字列にして FastEmbed に渡します。** FastEmbed は、英数字以外の除去やトークンの長さ制限を行います。英数字だけの文字列にして渡すことで、これらの影響を避けています。文書とクエリの両方に同じ変換をするので、同じ単語は同じトークンになります。
- **`Modifier.IDF` が必須です。** FastEmbed の BM25 はベクトルに TF の部分だけを入れ、IDF は Qdrant 側で計算します。
- **`avg_len` は実データから計算します。** FastEmbed の既定値は 256 です。短い文が中心のデータでは、文書長の補正がほとんど効かなくなります。

## 既知の課題

- **トークンの大文字小文字がそろっていません。** SudachiPy の `normalized_form()` は、同じ「Python」でも文脈によって `python` と `Python` を返します。両者は別のトークンとして扱われるため、検索で一致しないことがあります。tokenizer で `.lower()` をかけると解消できる見込みです。
- **`C++` は `c` になります。** SudachiPy が記号の `++` を落とすためです。`Next.js` は `ネクスト` と `js` に分かれます。
- **名詞以外は検索に使いません。**
- **否定は区別できません。** BM25 は語の出現だけを見るので、「経験あり」と「経験なし」は同じに扱われます。
- **文書が 6 件の動作確認です。** 検索精度は評価していません。

## 参考

- [Qdrant/fastembed `bm25.py`](https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm25.py)
- [SudachiPy](https://github.com/WorksApplications/SudachiPy)
