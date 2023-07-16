import os
import uuid
import requests
import textract
from urllib.parse import urlparse
import arxiv
import random
from datetime import datetime
from random_word import RandomWords
from gutenbergpy.gutenbergcache import GutenbergCache
import gutenbergpy.textget
import nltk
import csv
from halo import Halo
import argparse
from llama_cpp import Llama
from tqdm import tqdm
from quoters import Quote

nltk.download('punkt')

COLORS = ["\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m"]
RESET = "\033[0m"

SPINNER_STYLES = [
    'dots', 'dots2', 'dots3', 'dots4', 'dots5', 'dots6', 'dots7', 'dots8', 'dots9',
    'dots10', 'dots11', 'dots12', 'line', 'line2', 'pipe', 'simpleDots',
    'simpleDotsScrolling', 'star', 'star2', 'flip', 'hamburger', 'growVertical',
    'growHorizontal', 'balloon', 'balloon2', 'noise', 'bounce', 'boxBounce',
    'boxBounce2', 'triangle', 'arc', 'circle', 'squareCorners', 'circleQuarters',
    'circleHalves', 'squish', 'toggle', 'toggle2', 'toggle3', 'toggle4', 'toggle5',
    'toggle6', 'toggle7', 'toggle8', 'toggle9', 'toggle10', 'toggle11', 'toggle12',
    'toggle13'
]

SUPPORTED_EXTENSIONS = [
    ".csv", ".doc", ".docx", ".eml", ".epub", ".gif", ".htm", ".html", ".jpeg", 
    ".jpg", ".json", ".log", ".mp3", ".msg", ".odt", ".ogg", ".pdf", ".png", 
    ".pptx", ".ps", ".psv", ".rtf", ".tab", ".tff", ".tif", ".tiff", ".tsv", 
    ".txt", ".wav", ".xls", ".xlsx"
]

POTENTIAL_DELIMITERS = [
    ',', ';', '|', '\t', ':', ' ', 
    '#', '~', '^', '&', '*', '%', 
    '$', '@', '!', '?', '+', '-', 
    '=', '<', '>', '/', '\\', '`', 
    '[', ']', '{', '}', '(', ')', 
    '.', '_', '§', '°', '£', '€',
    "\uFFF9",  # Interlinear Annotation Separator
    "\uFFFA",  # Interlinear Annotation Anchor
    "\uFFFB",  # Interlinear Annotation Terminator
    "\uFFFC",  # Object Replacement Character
    "\uFFFD",  # Replacement Character
    "\uFEFF",  # Byte Order Mark
    "\u2063",  # Invisible Separator
    "\u2064",  # Invisible Plus
    "\u206A",  # Inhibit Symmetric Swapping
    "\u206B",  # Activate Symmetric Swapping
    "\u206C",  # Inhibit Arabic Form Shaping
    "\u206D",  # Activate Arabic Form Shaping
    "\u206E",  # National Digit Shapes
    "\u206F",   # Nominal Digit Shapes
]

def get_random_word(min_word_length):
    r = RandomWords()
    word_length = 0
    while word_length < min_word_length:
        word = r.get_random_word()
        if word:
            word_length = len(word)
    return word

# Search for papers on arXiv before a given date
def get_arxiv_article(beforedate="2018-01-01", searchterm=None):
    newlist = []
    randomsearch = False
    while len(newlist) < 1:
        if searchterm is None or randomsearch:
            randomsearch = True
            searchterm = get_random_word(random.randint(6, 12))

        dateobj = datetime.strptime(beforedate, "%Y-%m-%d")
        search = arxiv.Search(
            query=searchterm,
            max_results=100,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        # Get all results
        results = list(search.results())
        newlist = [result for result in results if result.updated.replace(tzinfo=None) < dateobj.replace(tzinfo=None)]
        if len(newlist) < 1:
            randomsearch = True

    return random.choice(newlist)

def decode_if_bytes(data):
    return data.decode() if isinstance(data, bytes) else data

def get_file_extension(path):
    return path.split("/")[-1].split('.')

def download_file(url, extension):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch webpage: {url}")
        return None

    filename = f"{uuid.uuid4()}.{extension}"
    with open(filename, 'wb') as file:
        file.write(response.content)
    
    return filename

def extract_text_from_file(filename):
    if not os.path.exists(filename):
        print(f"File does not exist: {filename}")
        return None

    text = textract.process(filename)
    os.remove(filename)
    return decode_if_bytes(text)

def get_text_from_document(path):
    parsed_path = urlparse(path)
    extension = get_file_extension(path)

    if extension not in SUPPORTED_EXTENSIONS:
        extension = "html"

    if parsed_path.netloc:
        # The path is a URL
        filename = download_file(path, extension)
    else:
        # The path is a local file
        filename = path

    return extract_text_from_file(filename) if filename else None

def find_delimiter(data):
    # Convert data to a single string
    data_str = str(data)

    # Find a character that's not in the data
    for delimiter in POTENTIAL_DELIMITERS:
        if delimiter not in data_str:
            return delimiter

    # If all potential delimiters are in the data, raise an error
    raise ValueError("Couldn't find a suitable delimiter")

def create_cache():
    GutenbergCache.create(refresh=True, download=True, unpack=True, parse=True, cache=True, deleteTemp=True)
    cache  = GutenbergCache.get_cache()
    return cache

def get_random_arxiv_text(beforedate="2018-01-01"):
    myarticle = get_arxiv_article(beforedate)
    articletext = get_text_from_document(myarticle.download_pdf())
    return articletext

def get_random_book_text(cache):
    rawtext = ""
    while len(rawtext) < 1:
        try:
            raw_book = gutenbergpy.textget.get_text_by_id(random.choice(cache.query(downloadtype=['application/plain','text/plain','text/html; charset=utf-8'], language=['en'])))
            rawtext = raw_book.decode()
            clean_book = gutenbergpy.textget.strip_headers(raw_book)
            text = clean_book.decode()
        except Exception as e:
            if "exceptions must derive from" in str(e):
                rawtext = ""
    return rawtext, text

def get_sentences(text):
    return nltk.sent_tokenize(text)

def group_sentences(sentences, group_size=5):
    return [" ".join(sentences[i:i+group_size]) for i in range(0, len(sentences), group_size)]

def generate_ai_output(inputtext, llm, tokens=128, inputprompt=None):
    output = None
    storedinputtext = inputtext
    tokenshrink=1
    while output is None:
        try:
            if inputprompt is None:
                inputprompt = "Q: Generate a different passage based on this passage style: "+inputtext+" A: "
            else:
                baseinputprompt=inputprompt.split(storedinputtext)[0]
                inputprompt=baseinputprompt+inputtext+" A: "
            output = llm(inputprompt, max_tokens=tokens, stop=["Q:"], echo=False)
        except Exception as e:
            if "tokens exceed" in str(e):
                inputtext = random.choice(storedinputtext.split('.'))
                inputtext = inputtext[:int(len(inputtext/tokenshrink))]
                tokenshrink+=1
                output = None
    return output['choices'][0]['text']

def write_to_csv(data, filename, delimiter=None):
    if delimiter is None:
        delimiter = find_delimiter(data)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=delimiter)
        writer.writerow(["classification", "text"])  # write header
        for item in data:
            for key, value in item.items():
                writer.writerow([key, value])  # write row

def generate_final_list(human_sent, ai_sent):
    returnlist = []
    for item in human_sent:
        if len(item) > 1:
            returnlist.append({0: item})
    for thing in ai_sent:
        if len(thing) > 1:
            returnlist.append({1: thing})
    random.shuffle(returnlist)
    return returnlist

def list_files(directory):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(directory) for file in files]


def main():
    parser = argparse.ArgumentParser(description='Generate a dataset for aidetector')
    parser.add_argument('--filename', type=str, help='The name of the file to write', required=True)
    parser.add_argument('--aigen', dest='aigen', action='store_true', help='Generate AI output to store in the dataset. If this flag is thrown, you must also specify a model using --model',required=False)
    parser.add_argument('--model', type=str, help='Specify a model to use with --aigen',required=False)
    parser.add_argument('--books', type=int, default=10, help='The number of gutenberg books to randomly sample from (default 10)',required=False)
    parser.add_argument('--quotes', dest='quotes', action='store_true', help='Pull quotes from the internet to provide something to read while AI is generating output.',required=False)
    parser.add_argument('--readbook', dest='readbook', action='store_true', help='Select one of the books at random and provide its story, sentence by sentence, providing something to read while the AI is generating text.',required=False)
    parser.add_argument('--modeldir',type=str,required=False,help="Specify the directory with compatible models if you are going to use multi-model generation")
    parser.add_argument('--nogutenberg',dest='nogutenberg',action='store_true',help='Specify this flag to prevent getting Project Gutenberg texts for human sources',required=False)
    parser.add_argument('--noarxiv',dest='noarxiv',action='store_true',help='Specify this flag to prevent getting arxiv texts for human sources',required=False)

    args = parser.parse_args()

    if args.aigen and (args.model is None and args.modeldir is None):
        parser.error("--model or --modeldir is required when --aigen is used")
    if args.model and args.modeldir:
        parser.error("You can only specify a single model or a directory filled with models to generate text from.")
    if args.readbook and args.quotes:
        args.quotes=False

    modellist = list_files(args.modeldir) if args.modeldir else []
    booklist = []
    finalsentgroup = []
    aisentgroup = []

    with Halo(text="Generating dataset...", spinner=random.choice(SPINNER_STYLES)) as spinner:
        if not args.nogutenberg:
            cache = create_cache()
            for i in range(args.books):
                spinner.text = f"Processing book {i}"
                spinner.spinner = random.choice(SPINNER_STYLES)
                rawtext, text = get_random_book_text(cache)
                currentsentences = get_sentences(text)
                if args.readbook:
                    booklist.extend(currentsentences)
                finalsentgroup.extend(group_sentences(currentsentences))

        if not args.noarxiv:
            for i in range(args.books):
                spinner.text = f"Processing article {i}"
                spinner.spinner = random.choice(SPINNER_STYLES)
                text = get_random_arxiv_text()
                currentsentences = get_sentences(text)
                if args.readbook:
                    booklist.extend(currentsentences)
                finalsentgroup.extend(group_sentences(currentsentences))

        booklist.reverse()

        if args.aigen:
            llm = Llama(model_path=args.model, verbose=False) if args.model else None
            pbar = tqdm(finalsentgroup, desc="Generating AI Output")
            for i, sent in enumerate(pbar):
                if len(modellist) > 0:
                    while llm is None:
                        modelpath = random.choice(modellist)
                        try:
                            llm = Llama(model_path=modelpath, verbose=False)
                        except Exception as e:
                            if "really a GGML file" in str(e):
                                llm = None
                                os.remove(modelpath)

                mq = "Generating AI Output..."
                if args.quotes:
                    mq = Quote.print()
                elif args.readbook:
                    mq = booklist.pop()

                color = COLORS[i % len(COLORS)]
                pbar.set_description(f"{color}{mq}{RESET}")
                spinner.text = "Generating AI portion of dataset... "
                spinner.spinner = random.choice(SPINNER_STYLES)
                aisentgroup.append(generate_ai_output(sent, llm))

        spinner.text = "Generating final list..."
        spinner.spinner = random.choice(SPINNER_STYLES)
        finallist = generate_final_list(finalsentgroup, aisentgroup) 
        write_to_csv(finallist, args.filename)

if __name__ == "__main__":
    main()

