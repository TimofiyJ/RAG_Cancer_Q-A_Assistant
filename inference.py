from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
import os
from dotenv import load_dotenv
import dspy

load_dotenv(override=True)

class GenerateAnswer(dspy.Signature):
    """With the provided context answer the question about cancer, as medical condition, mostly based on the information you have been provided with. Answer has to be
    accurate and consistent. There should be no deviation from the topic of cancer. You are responsible and accurate assistant
    so remember no joke or anecdotes and answer questions in formal maner just about cancer."""

    context = dspy.InputField(
        desc="Accurate medical information that the response should be based on."
    )
    question = dspy.InputField()

    answer = dspy.OutputField(
        desc="Full answer with finished thought that ends with dot symbol. Don't include reasoning and question in you answer"
    )


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HUGGING_FACE_API_KEY"),
            model_name=os.getenv("RETREIVAL_MODEL_NAME"),
        )
        self.retrieve = ChromadbRM(
            "cancer",
            persist_directory=os.path.join("vector_db","cancer"),
            embedding_function=self.huggingface_ef,
            k=2,
        )
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def main():

    client = chromadb.PersistentClient(os.path.join("vector_db","cancer"))
    collection = client.get_or_create_collection(
        name="cancer"
    )  # create collection if it doesn't exist

    model = dspy.GROQ(
        model=os.getenv("LLM_NAME"), api_key=os.environ.get("GROQ_API_KEY")
    )

    client = chromadb.PersistentClient(os.path.join("vector_db","cancer"))

    collection = client.get_or_create_collection(
        name="cancer"
    )  
    dspy.settings.configure(lm=model)
    question = input("What is your question?\n")

    rag = RAG()

    pred = rag(question)

    print(f"Question: {question}")
    print(f"Predicted Answer: {pred.answer}")


if __name__ == "__main__":
    main()