import pandas as pd
import sqlite3
from sqlite3 import Error
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
df = pd.read_csv('test.txt', sep='\t')

df = df[["ArticleTitle","Question"]]

df.rename(columns={"Question": "content"}, inplace=True)

df.dropna(axis=0, inplace=True)

print(df.head())

#create_connection(r"D:\Machine Learning\NLP\Haystack\faissIndex.db")

document_store_faiss = FAISSDocumentStore(faiss_index_factory_str="Flat")

retriever_faiss = EmbeddingRetriever(document_store=document_store_faiss,
                                     embedding_model='distilroberta-base-msmarco-v2',
                                     model_format='sentence_transformers', use_gpu=False)

document_store_faiss.write_documents(df.to_dict(orient='records'))

document_store_faiss.update_embeddings(retriever=retriever_faiss)

print(retriever_faiss.retrieve("Did his mother die of pneumonia?",top_k=10))




