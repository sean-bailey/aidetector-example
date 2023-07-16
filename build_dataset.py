"""
In here we just want to build the dataset for human input.

"""

from gutenbergpy.gutenbergcache import GutenbergCache
import random
import gutenbergpy.textget
import nltk
import csv
from halo import Halo
import argparse
nltk.download('punkt')
from llama_cpp import Llama
from tqdm import tqdm
from quoters import Quote
import os


colors = ["\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m"]
RESET = "\033[0m"


spinner_styles = [
    'dots', 'dots2', 'dots3', 'dots4', 'dots5', 'dots6', 'dots7', 'dots8', 'dots9',
    'dots10', 'dots11', 'dots12', 'line', 'line2', 'pipe', 'simpleDots',
    'simpleDotsScrolling', 'star', 'star2', 'flip', 'hamburger', 'growVertical',
    'growHorizontal', 'balloon', 'balloon2', 'noise', 'bounce', 'boxBounce',
    'boxBounce2', 'triangle', 'arc', 'circle', 'squareCorners', 'circleQuarters',
    'circleHalves', 'squish', 'toggle', 'toggle2', 'toggle3', 'toggle4', 'toggle5',
    'toggle6', 'toggle7', 'toggle8', 'toggle9', 'toggle10', 'toggle11', 'toggle12',
    'toggle13'
]

potential_delimiters = [
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


def find_delimiter(data):
    
    # Convert data to a single string
    data_str = str(data)

    # Find a character that's not in the data
    for delimiter in potential_delimiters:
        if delimiter not in data_str:
            return delimiter

    # If all potential delimiters are in the data, raise an error
    raise ValueError("Couldn't find a suitable delimiter")



def createCache():
    GutenbergCache.create(refresh=True, download=True, unpack=True, parse=True, cache=True, deleteTemp=True)
    cache  = GutenbergCache.get_cache()
    return cache

def getRandomBookText(cache):
    rawtext=""
    while len(rawtext)<1:
        try:
            raw_book=gutenbergpy.textget.get_text_by_id(random.choice(cache.query(downloadtype=['application/plain','text/plain','text/html; charset=utf-8'], language=['en'])))
            rawtext=raw_book.decode()
            rawtext=raw_book.decode()
            clean_book = gutenbergpy.textget.strip_headers(raw_book)
            text=clean_book.decode()
        except Exception as e:
            if "exceptions must derive from" in str(e):
                rawtext=""
    return rawtext,text

def getSentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def group_sentences(sentences, group_size=5):
    groups = []
    for i in range(0, len(sentences), group_size):
        group = " ".join(sentences[i:i+group_size])
        groups.append(group)
    return groups

def generateAIOutput(inputtext,llm,tokens=128,inputprompt=None):
    output=None
    storedinputtext=inputtext
    while output==None:
        try:
            if inputprompt is None:
                inputprompt="Q: Determine the style of this passage and write a different passage in a similar style. Do not describe the style or the passage, provide your passage only: "+inputtext+" A: "
            output = llm(inputprompt, max_tokens=tokens, stop=["Q:"], echo=False)
        except Exception as e:
            if "tokens exceed" in str(e):
                inputtext=random.choice(storedinputtext.split('.'))
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
#make sure the dataset puts 1 next to AI Generated text, and 0 next to human generated text.
def generateFinalList(humansent,aisent):
    returnlist=[]
    for item in humansent:
        if len(item)>1:
            returnlist.append({0:item})
    for thing in aisent:
        if len(thing)>1:
            returnlist.append({1:thing})
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

    parser.set_defaults(aigen=False)
    parser.set_defaults(quotes=False)
    parser.set_defaults(readbook=False)


    args = parser.parse_args()

    if args.aigen and (args.model is None and args.modeldir is None):
        parser.error("--model or --modeldir is required when --aigen is used")
    if args.model and args.modeldir:
        parser.error("You can only specify a single model or a directory filled with models to generate text from.")
    if args.readbook and args.quotes:
        args.quotes=False
    modellist=[]
    if args.modeldir:
        modellist=list_files(args.modeldir)
    with Halo(text="Generating dataset...", spinner=random.choice(spinner_styles)) as spinner:
        #We need to get the groups
        finalsentgroup=[]
        booklist=[]
        cache=createCache()
        for i in range(args.books):
            spinner.text="Processing book "+str(i)
            spinner.spinner=random.choice(spinner_styles)
            rawtext,text=getRandomBookText(cache)
            currentsentences=getSentences(text)
            if args.readbook:
                for sent in currentsentences:
                    booklist.append(sent)
            sentgroup=group_sentences(currentsentences)
            for group in sentgroup:
                finalsentgroup.append(group)
        #now, we need to create AI output based on this final sentence group.
        aisentgroup=[]
        booklist.reverse()
        if args.aigen:
            llm=None
            if args.model:
                llm = Llama(model_path=args.model,verbose=False)
            pbar = tqdm(finalsentgroup, desc="Generating AI Output")
            for i, sent in enumerate(pbar):
                #we want to generate the same _amount_ of AI generated responses, but we want to get as many different samples from as many different models as possible.
                if len(modellist)>0:
                    while llm==None:
                        #just in case we run into incompatible llms...
                        modelpath=random.choice(modellist)
                        try:
                            llm=Llama(model_path=modelpath,verbose=False,)
                        except Exception as e:
                            if "really a GGML file" in str(e):
                                llm=None
                                os.remove(modelpath)
                        
                mq="Generating AI Output..."
                if args.quotes:
                    mq=Quote.print()
                elif args.readbook:
                    mq=booklist.pop()
                
                color = colors[i % len(colors)]
                pbar.set_description(f"{color}{mq}{RESET}")
                spinner.text="Generating AI portion of dataset... "
                spinner.spinner = random.choice(spinner_styles)
                aisentgroup.append(generateAIOutput(sent,llm))
        spinner.text="Generating final list..."
        spinner.spinner=random.choice(spinner_styles)
        finallist=generateFinalList(finalsentgroup,aisentgroup) 
        write_to_csv(finallist, args.filename)


if __name__ == "__main__":
    main()
