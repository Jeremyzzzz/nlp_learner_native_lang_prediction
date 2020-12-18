from bs4 import BeautifulSoup
from zipfile import ZipFile


def corpus_reader(zipfile):
    """A generator yield filename, language, and main text each time from the zipfile

    Args:
        zipfile (Zipfile): A Zipfile instance pointing to a zip file

    Yields:
        Tuple: filename, language, and main text (tuple(str, str, str))
    """
    for filename in zipfile.namelist():
        if filename.endswith(".html"):
            with zipfile.open(filename) as f:
                doc = filename.split("/")[1]
                html = BeautifulSoup(f.read(), "lxml")
                native_lang = get_L1(html)
                body_text = get_text(html)
                yield doc, native_lang, body_text


def get_text(soup):
    """Return the main body text from given soup

    Args:
        soup (bs4.BeautifulSoup): A Beautiful soup instance of html

    Returns:
        str: Returns the main body text from the soup
    """
    for div in soup.find_all("div"):
        if "id" in div.attrs and div.attrs["id"] == "body_show_ori":
            text = div.get_text()
    return text


def get_L1(soup):
    """Return the native speaking language of author from given soup

    Args:
        soup (bs4.BeautifulSoup): A Beautiful soup instance of html

    Returns:
        str: Returns native language spoken from the soup
    """

    return soup.find(
        "li", recursive=True, attrs={"title": "Native language"}
    ).get_text()


def get_filename(filepath):
    """Return file name based on the file path.

    Args:
        filepath (str): A filepath deliminated with /

    Returns:
        str: Returns the name of the file
    """

    return filepath.split("/")[-1]