import os
from pprint import pprint

from haystack.nodes import EmbeddingRetriever
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.utils import print_answers



doc_dir = "data/build_your_first_question_answering_system"

fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
    output_dir=doc_dir
)

document_store_faiss = FAISSDocumentStore(faiss_index_factory_str="Flat")

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store_faiss)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever_faiss = EmbeddingRetriever(
    document_store=document_store_faiss,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
)

document_store_faiss.update_embeddings(retriever_faiss)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipe = ExtractiveQAPipeline(reader, retriever_faiss)


prediction = pipe.run(
    query="Who is the father of Arya Stark?",
    params={
        "Retriever": {"top_k": 10},
        "Reader": {"top_k": 5}
    }
)

pprint(prediction)

print_answers(
    prediction,
    details="minimum" ## Choose from `minimum`, `medium`, and `all`
)



