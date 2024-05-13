from sentence_transformers import SentenceTransformer
import chromadb
import os
import uuid
import vector_db.data_preparation as data_preparation
from chromadb import Collection
from dotenv import load_dotenv

load_dotenv()


def insert_record(model: SentenceTransformer, collection: Collection, info: str):
    """Insert record into collection (vectordb)

    Args:
        model : model for encoding
        collection : target collection
        info : main information
    """
    query_vector = model.encode(info).tolist()

    collection.add(
        embeddings=[query_vector],
        documents=[info],
        metadatas=[{"metadata": "1"}],  # TODO: expand information in metadata
        ids=[str(uuid.uuid4())],
    )


def load_data():
    """Gets clean data, transform it into documents and then into nodes,
    insert data into vectordb
    """
    documents = data_preparation.get_clean_documents(os.listdir("../data"))
    nodes = data_preparation.get_nodes(documents)

    client = chromadb.PersistentClient(path="./cancer")

    collection = client.get_or_create_collection(
        "cancer", metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(os.getenv("RETREIVAL_MODEL_NAME")).to("cuda")

    for idx, line in enumerate(nodes):
        try:
            insert_record(
                model=model,
                collection=collection,
                info=line.text,
            )
        except Exception as e:
            with open("bad_lines.txt", "a") as errors_f:
                errors_f.write(f"{line} error is: {e}\n")


if __name__ == "__main__":
    load_data()
