import requests
from bs4 import BeautifulSoup
import logging
import os

logger = logging.getLogger(__name__)


def get_cancer_types_url(url: str):
    """Extract all the cancer types from the given site(url).
    Parses url(sitemap of the site) and appends pages of each type of cancer
    that is expected to have pdf files inside.
    Args:
        url (string): sitemap of the site(usually xml) that contains all the routes of the site

    Raises:
        Exception: When there is error connecting to the site there is no need of the future steps
        so the programs end with the error message

    Returns:
        list: list that contain type of cancer own url of the page
    """
    logger.info(f"Sent request to {url}")

    try:
        r = requests.get(url)
    except Exception as e:
        logger.info(f"Error in request to {url}: {e}")
        print(f"Error in request to {url}: {e}")
        return 1

    if r.status_code != 200:
        logger.info(f"Status code for {url} is {r.status_code}")
        raise Exception(f"Status code for {url} is {r.status_code}")

    logger.info(f"Request code to {url} is 200")

    soup = BeautifulSoup(r.text, "xml")
    url_list = []

    for link_section in soup.find_all("url"):
        try:
            link_text = link_section.find("loc").text
            if link_text.startswith(
                "https://www.cancer.org/cancer/types/"
            ):  # contains list of types of cancer
                if len(link_text.split("/")) == 6:  # next level after types
                    url_list.append(link_text)
        except Exception as e:
            logger.info(f"Error in parsing link {link_section}: {e}")
            print(f"Error in parsing link {link_section}: {e}")
    return url_list


def extract_pdf(url: str):
    """Extracts all the pdf links in a given page to the list
    Args:
        url (string): url of the page which is needed to be parsed and scraped for pdf link
    Returns:
        list: list of links to the pages with pdf
    """
    logger.info(f"Sent request to {url}")

    try:
        r = requests.get(url)
    except Exception as e:
        logger.error(f"Error in request to {url}: {e}")
        print(f"Error in request to {url}: {e}")
        return None

    if r.status_code != 200:
        logger.warning(f"Status code for {url} is {r.status_code}")
        print(f"Status code for {url} is {r.status_code}")
        return None

    logger.info(f"Request code to {url} is 200")

    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("a")
    pdfs = []

    for link in links:
        link = link.get("href")
        try:
            if link.endswith(".pdf"):
                pdfs.append("https://www.cancer.org/" + link)  # append full url
        except Exception as e:
            logger.error(f"Error in extracting pdf {link}: {e}")
            print(f"Error in extracting pdf {link}: {e}")
    return pdfs


def download_pdf(url: str, download_path: str):
    """Sends request and downloads the content (pdf) of the page.
    Saves pdf file in 'download_path' folder and names it after the url.
    Args:
        url (string): the url of the target pdf file
        download_path (string): the folder for saving the pdf file

    Returns:
        None: None if there was an error
    """

    logger.info(f"Sent request to {url}")

    try:
        r = requests.get(url)
    except Exception as e:
        logger.error(f"Error in request to {url}: {e}")
        print(f"Error in request to {url}: {e}")
        return None

    if r.status_code != 200:
        logger.warning(f"Status code for {url} is {r.status_code}")
        print(f"Status code for {url} is {r.status_code}")
        return None

    logger.info(f"Request code to {url} is 200")
    logger.info("Downloading pdf")

    filename = url.replace("/", "")  # replaces / so it can be saved with no errors
    filename = filename.replace(":", "")  # replaces : so it can be saved with no errors

    download_file = os.path.join(download_path, filename)

    with open(download_file, "wb") as f:
        f.write(r.content)


def main():
    main_folder = os.getcwd()
    parser_folder = os.path.join(main_folder, "parser")
    data_folder = os.path.join(main_folder, "data")

    logger_path = os.path.join(parser_folder, "parser.log")

    logging.basicConfig(
        filename=logger_path,
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(message)s",
    )

    url = "https://www.cancer.org/sitemap.xml"  # page with all the routes of the site
    cancer_types_urls = get_cancer_types_url(url)

    if len(cancer_types_urls) == 0:
        logger.warning("There are 0 cancer type urls")
        return 0

    for cancer_type_url in cancer_types_urls:
        pdfs = extract_pdf(cancer_type_url)
        if pdfs is None:
            logger.warning(f"Didn't extract pdf from {cancer_type_url}")
            print(f"Didn't extract pdf from {cancer_type_url}")
        for pdf_url in pdfs:
            download_pdf(pdf_url, data_folder)

    logger.info("Finished parsing")
    print("Finished parsing")


if __name__ == "__main__":
    main()
