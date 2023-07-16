"""

This is going to be a bit clever.

First, we'll need to pull in the human bits from gutenberg. 

We will run aidetector against the human incoming text, and check to see if it 
correctly classifies it. If it does, no need to record it. If it does not, add it to the new dataset.

Generate an AI output based on this incoming text. Check if it's properly classified. If it is, run it through a loop
of regeneration with a prompt like "an AI detection classfier previously determined this (x) was written by AI, rewrite it so that it won't be detected."
Once the classifier fails, take that AI output and add it to the dataset. 

Once this new dataset reaches an arbitrary length (5000 entries?) it will then fine tune the model, and add this dataset to an old, overall dataset.

Scratch that, One big old overall dataset for availability, with the smaller dataset just kept in memory.

Keep the last 5 fine tuned models, with their overall accuracy scores in their names.

It would also be nifty if we could have this thing running continuously in the background, with some sort of separate server interface which allows you to pull models,
datasets, etc.
"""

from build_dataset import *
from aidetector.tokenization import *
from aidetector.inference import *
from aidetector.aidetectorclass import *
from aidetector.training import *
import uuid
import os
import glob
import shutil
import subprocess
import pandas as pd
import torch
from halo import Halo
from llama_cpp import Llama
import random
import csv

DELIMITER = "\u2064"
EVAL_THRESHOLD = 0.98

def return_specific_files(directory, extension):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(directory) 
            for file in files if os.path.splitext(file)[1] == extension]

def move_file(source_filepath, destination_directory):
    shutil.move(source_filepath, destination_directory)

def test_phrase(input_text, model_file, vocab_file):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    vocab = load_vocab(vocab_file)
    model = AiDetector(len(vocab))
    model.load_state_dict(torch.load(model_file))
    is_ai = check_input(model, vocab, input_text, tokenizer=tokenizer)
    return is_ai

def append_to_csv(filename, data, delimiter):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        if not file_exists:
            writer.writerow(["classification", "text"])
        writer.writerow(data)

def random_boolean():
    return random.choice([True, False])

def fine_tune_model(input_model, input_dataset, output_model_file, output_vocab_file, percent_split=0.2, classification_label="classification", text_label="text", epochs=100, lower_bound=0.3, upper_bound=0.5):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    train_txt, test_text, train_labels, test_labels = load_data(input_dataset, percent_split=percent_split, classification_label=classification_label, text_label=text_label)
    vocab, train_seqs, test_seqs = tokenize_data(train_txt, test_text, tokenizer)
    model = AiDetector(len(vocab))
    model.load_state_dict(torch.load(input_model))
    _ = model(torch.zeros(1, train_seqs.size(1)).long())
    model.add_fc_layer()
    spinner = Halo(text='Fine-tuning model', spinner='dots')
    spinner.start()
    train_model(model, train_seqs, torch.tensor(train_labels.values, dtype=torch.float), test_seqs, torch.tensor(test_labels.values, dtype=torch.float), epochs=epochs, lower_bound=lower_bound, upper_bound=upper_bound)
    spinner.stop()
    save_model(model, output_model_file)
    save_vocab(vocab=vocab, vocab_output_file=output_vocab_file)

def list_files_by_age(directory, extension):
    files = glob.glob(f"{directory}/*.{extension}")
    files.sort(key=os.path.getmtime)
    return files

def eval_model(test_file, vocab_file, model_file):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    vocab = load_vocab(vocab_file)
    model = AiDetector(len(vocab))
    df = pd.read_csv(test_file, sep=detect_delimiter(test_file, "classification", "text"))
    accuracy_sum = 0
    accuracy_count = 0
    for index, row in df.iterrows():
        classification = row['classification']
        text = str(row['text'])
        if len(text) > 0:
            try:
                model.load_state_dict(torch.load(model_file))
                is_ai = check_input(model, vocab, text, tokenizer=tokenizer)
                if is_ai == classification:
                    accuracy_sum += 1
                accuracy_count += 1
            except Exception as e:
                print(text)
                print(e)
    evaluation_percent = accuracy_sum / accuracy_count
    return evaluation_percent

def main():
    parser = argparse.ArgumentParser(description='Continuously generate a dataset and model for aidetector')
    parser.add_argument('--currentdir', type=str, help='The directory for the current model/vocab', required=True)
    parser.add_argument('--olddir', type=str, default="olddir", help='The old file directory', required=False)
    parser.add_argument('--fulldataset',type=str,default='fulldataset.csv',help='The full total dataset',required=False)
    parser.add_argument('--modeldir',type=str,required=True,help="Specify the directory with compatible models")
    parser.add_argument('--traincounter',type=int,required=False,default=5000,help="The size of the next iteration of the dataset to fine tune the model with.")
    parser.add_argument("--archivecount",type=int,required=False,default=5,help="The number of older models and vocab files to keep")
    parser.add_argument("--evaldataset",type=str,required=False,default=None,help="The premade evaluation dataset")
    args = parser.parse_args()
    current_model = return_specific_files(args.currentdir, ".aidetectormodel")[0]
    current_vocab = return_specific_files(args.currentdir, ".vocab")[0]
    try:
        current_csv = return_specific_files(args.currentdir, ".csv")[0]
    except:
        current_csv = os.path.join(args.currentdir, str(uuid.uuid4()).replace("-", "") + ".csv")
    book_cache = create_cache()
    current_directory = os.getcwd()
    old_dir = os.path.join(current_directory, args.olddir)
    full_dataset = os.path.join(old_dir, args.fulldataset)
    eval_dataset = None
    if args.evaldataset is None:
        eval_dataset = os.path.join(args.currentdir, str(uuid.uuid4()).replace("-", "") + ".csv")
        cmd = ['python3', 'build_dataset.py', "--filename", eval_dataset, "--aigen", "--modeldir", args.modeldir]
        subprocess.run(cmd)
    else:
        eval_dataset = args.evaldataset
    current_eval_percent = eval_model(eval_dataset, vocab_file=current_vocab, model_file=current_model)
    os.makedirs(old_dir, exist_ok=True)
    current_counter = 0
    with Halo(text="Generating dataset...", spinner=random.choice(SPINNER_STYLES)) as spinner:
        while True:
            if random_boolean():
                raw_text, text = get_random_book_text(book_cache)
            else:
                text=get_random_arxiv_text()
            sentences = get_sentences(text)
            sent_groups = group_sentences(sentences)
            for sent in sent_groups:
                spinner.text = "Iterating through the human phrases..."
                spinner.spinner = random.choice(SPINNER_STYLES)
                is_ai = test_phrase(sent, current_model, current_vocab)
                if is_ai == 1:
                    spinner.text = "Got a human made phrase the model thought was AI!"
                    spinner.spinner = random.choice(SPINNER_STYLES)
                    data = [0, sent]
                    append_to_csv(current_csv, data, delimiter=DELIMITER)
                    append_to_csv(full_dataset, data, delimiter=DELIMITER)
                    current_counter += 1
                current_llm_list = list_files(args.modeldir)
                llm = None
                while llm is None:
                    spinner.text = "Selecting an LLM to generate the text with..."
                    spinner.spinner = random.choice(SPINNER_STYLES)
                    model_path = random.choice(current_llm_list)
                    try:
                        llm = Llama(model_path=model_path, verbose=False)
                    except Exception as e:
                        if "really a GGML file" in str(e):
                            llm = None
                            os.remove(model_path)
                output_text = generate_ai_output(sent, llm)
                while test_phrase(output_text, current_model, current_vocab) == 0:
                    spinner.text = "Fooled the classifier model! Generated an AI output which looks 'human' enough..."
                    spinner.spinner = random.choice(SPINNER_STYLES)
                    data = [1, output_text]
                    append_to_csv(current_csv, data, delimiter=DELIMITER)
                    append_to_csv(full_dataset, data, delimiter=DELIMITER)
                    current_counter += 1
                    output_text = generate_ai_output(sent, llm, input_prompt="Q: This text was determined to be AI Generated: "+output_text+" Rewrite it to not be detectable as AI. A:")
            if current_counter >= args.traincounter:
                spinner.text = "Looks like we have enough data now, fine-tuning..."
                spinner.spinner = random.choice(SPINNER_STYLES)
                new_name = str(uuid.uuid4()).replace("-", "")
                new_model = os.path.join(args.currentdir, new_name + ".aidetectormodel")
                new_vocab = os.path.join(args.currentdir, new_name + ".vocab")
                fine_tune_model(input_model=current_model, input_dataset=current_csv, output_model_file=new_model, output_vocab_file=new_vocab)
                new_eval_percent = eval_model(eval_dataset, vocab_file=new_vocab, model_file=new_model)
                if new_eval_percent >= current_eval_percent:
                    spinner.text = "The new classifier performed better!"
                    spinner.spinner = random.choice(SPINNER_STYLES)
                    current_eval_percent = new_eval_percent
                    move_file(current_model, old_dir)
                    move_file(current_vocab, old_dir)
                    old_model_list = list_files_by_age(old_dir, ".aidetectormodel")
                    old_vocab_list = list_files_by_age(old_dir, ".vocab")
                    old_model_list.reverse()
                    old_vocab_list.reverse()
                    while len(old_model_list) > args.archivecount:
                        os.remove(old_model_list.pop())
                    while len(old_vocab_list) > args.archivecount:
                        os.remove(old_vocab_list.pop())
                    os.remove(current_csv)
                    current_csv = os.path.join(args.currentdir, str(uuid.uuid4()).replace("-", "") + ".csv")
                    current_model = new_model
                    current_vocab = new_vocab
                    while current_eval_percent > EVAL_THRESHOLD:
                        spinner.text = "Our model is performing higher than our threshold. Lets see if we can generate better data to stump it..."
                        spinner.spinner = random.choice(SPINNER_STYLES)
                        cmd = ['python3', 'build_dataset.py', "--filename", eval_dataset, "--aigen", "--modeldir", args.modeldir]
                        subprocess.run(cmd)
                        current_eval_percent = eval_model(eval_dataset, vocab_file=current_vocab, model_file=current_model)
                else:
                    spinner.text = "This new model doesn't perform better than our current model. Let's trash it and try again."
                    spinner.spinner = random.choice(SPINNER_STYLES)
                    os.remove(new_model)
                    os.remove(new_vocab)
                current_counter = 0

if __name__ == "__main__":
    main()

