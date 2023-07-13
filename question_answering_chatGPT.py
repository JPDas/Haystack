import logging
from haystack.utils import fetch_archive_from_http
from haystack.utils import convert_files_to_docs, clean_wiki_text
from haystack.nodes import PreProcessor
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import OpenAIAnswerGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


# This fetches some sample files to work with
doc_dir = "data/build_your_first_question_answering_system"
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

preprocessor = PreProcessor(
 clean_empty_lines=True,
 clean_whitespace=True,
 clean_header_footer=False,
 split_by="word",
 split_length=100,
 split_overlap=3,
 split_respect_sentence_boundary=False,
)

processed_docs = preprocessor.process(docs)

print(processed_docs[0])
print('*********************************************************')


document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", embedding_dim=1536)

document_store.delete_documents()
document_store.write_documents(processed_docs)

print("Updated document store")
MY_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxx"
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="text-embedding-ada-002",
    batch_size = 32,
    api_key=MY_API_KEY,
    max_seq_len = 1024,
    use_gpu = False
)

document_store.update_embeddings(retriever)

print("Calling OpenAIAnswer")
generator = OpenAIAnswerGenerator(api_key=MY_API_KEY, model="text-davinci-003", temperature=0, max_tokens=300)

print("Generating pipeline")
gpt_search_engine = GenerativeQAPipeline(generator=generator, retriever=retriever)

answer = gpt_search_engine.run(
    query="Who is the father of Arya Stark?",
    params={
        "Retriever": {"top_k": 5},
        "Generator": {"top_k": 1, "timeout": 300}
    }
)

print_answers(answer, details="minimum")
