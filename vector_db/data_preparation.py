from llama_index.readers.file import PDFReader
import re
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from typing import List
from dotenv import load_dotenv
from llama_index.core.node_parser import SemanticSplitterNodeParser

load_dotenv()


def remove_underlines(text: str):
    return re.sub("_+", "", text)


def get_dot_sentences(text: str):
    """Mostly sentences marked with ● are either internal links or paragraphs.
    In these documents usually firstly comes internal link that teleports to the paragraph so
    we remove them since it doesn't contain much sence and usually ruins structure of document text
    Args:
        text (str): text to detect dot sentences from

    Returns:
        list: list of sentences which end with ●
    """
    sentences = []
    for l in range(len(text.split("\n"))):
        line = text.split("\n")[l]
        if "●" in line:
            index = line.index("●")
            sentence = line[: index + 1].strip()

            if sentence[0].isupper():
                sentences.append(sentence)
                continue

            adding_line = l
            while adding_line != 0:
                if text.split("\n")[adding_line][0].isupper() is not True:
                    sentence = text.split("\n")[adding_line - 1] + " " + sentence
                else:
                    break
                adding_line -= 1
            sentences.append(sentence)

    return sentences


def remove_inner_links(documents_loader: List[Document]):
    """Removes inner links since they don't provide useful information
    and ruin structure of document. documents_loader allows analyse all pages in
    sequential order
    Args:
        documents_loader : object that contains all pages of the text
    """
    for i in range(len(documents_loader)):
        page = documents_loader[i]
        sentences = get_dot_sentences(page.text)

        for sentence in sentences:
            find_sentence = sentence
            index = sentence.find("●")
            if index != -1:
                find_sentence = sentence[:index]
            delete_flag = False
            if page.text.count(find_sentence) > 1:
                delete_flag = True
            else:
                for j in range(i + 1, len(documents_loader)):
                    checking_page = documents_loader[j].text.replace("\n", " ")
                    if find_sentence in checking_page:
                        delete_flag = True
                        break
            if delete_flag:
                page.text = page.text.replace(sentence, "", 1)


def writtenby_to_end(text: str):
    """Each document has written by section that is located at the end of the document
    and finishes with link of the organization (1.800.227.2345)

    Args:
        text (str): text of the document

    Returns:
        paragraph of written by section
    """
    match = re.search(r"Written by.*?(1.800.227.2345)", text, re.DOTALL)
    if match:
        return match.group(0)
    return ""


def hyperlinks_to_references(text: str):
    """Each page consists of hyperlinks and references section that has no meaning for our data
    so we remove them. These section begin with Hyperlinks and end with Last Revised and data

    Args:
        text (str): text of the page

    Returns:
        paragraph of the sections that are needed to be removed
    """
    return re.sub(
        r"Hyperlinks.*?Last Revised:.*?(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})",
        "\n",
        text,
        flags=re.DOTALL,
    )


def remove_page_number(page: Document, number: int):
    text = page.text
    return re.sub(r"(\s*){}\s*$".format(number), "", text)


def get_clean_documents(data_folder: List[str]):
    """Extract all text information from pdf files located in data folder.
    Clean the text for each document page and document as a whole.
    Append document text as plain text to text_list, then
    translate it to Document object
    Args:
        data_folder (list[str]): list of paths of all data

    Returns:
        list: Array of objects Document type
    """
    text_list = []

    for document_name in data_folder:
        loader = PDFReader()

        try:
            documents_loader = loader.load_data(file="./data/" + document_name)
        except Exception as e:
            print("Cant read file, error: ", e)
            continue

        remove_inner_links(documents_loader)

        document_text = ""

        for document_number in range(len(documents_loader)):
            documents_loader[document_number].text = remove_page_number(
                documents_loader[document_number], document_number + 1
            )
            document_text = (
                document_text + documents_loader[document_number].get_content()
            )

        document_text = hyperlinks_to_references(document_text)
        document_text = document_text.replace(writtenby_to_end(document_text), "")
        document_text = document_text.replace(
            "American Cancer Society cancer.org | 1.800.227.2345 ____________________________________________________________________________________",
            "",
        )
        document_text = document_text.replace("cancer.org | 1.800.227.2345", "")
        document_text = remove_underlines(document_text)
        document_text = document_text.strip()
        document_text = re.sub(r"\n{2,}", "\n", document_text)
        document_text = re.sub(r"[^\x00-\x7F]+", "", document_text)

        text_list.append(document_text)

    documents = [Document(text=t) for t in text_list]

    return documents


def get_nodes(documents: List[Document]):
    """Perform semantic chunking on the documents and transform them to nodes
    Args:
        documents (list): list of objects Document type that are to be converted to Node type

    Returns:
        list: array of objects Node type
    """
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=HuggingFaceEmbedding(model_name=os.getenv("RETREIVAL_MODEL_NAME")),
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes
