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
import time
import os
import glob
import shutil
import subprocess


DELIMITER="\u2064"
evalthreshold=0.98


#We're going to need this thing to have a bit of organization. It needs to intelligently auto-run effectively forever. 
#This means it will have an "old" and "current" folder structure.

#To start this, we'll have to specify a "current" directory, and optionally an "old" directory. If old is not specified, it will be made.
#This thing will pull the current aidetector model and vocab from the current directory. Once it needs re-training, the current model gets dumped into the old directory
#with some form of labeling system, with the new model being put in the current directory. It gets reloaded every time from the current directory. 
#We can probably pull off using the current/old locations for our datasets too, though we'll be mostly just using old.
#keep default 5 old models, but allow user to specify how many models to keep. I'm completely unconcerned about the size of the text. Gutenberg maxes out at what, 300GB?
#We'll expand from there. PoC.



#I'm going to enforce some norms here. I want the aidetector model file to end in .aidetectormodel, and the vocab to end in .vocab. I'll tell the user this if not detected.
def return_specific_files(directory, extension):
    return [os.path.join(dirpath, file) for dirpath, dirnames, files in os.walk(directory) 
            for file in files if os.path.splitext(file)[1] == extension]

def move_file(source_filepath, destination_directory):
    shutil.move(source_filepath, destination_directory)

def testPhrase(inputtext,modelfile,vocabfile):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    vocab=load_vocab(vocabfile)
    model=AiDetector(len(vocab))
    model.load_state_dict(torch.load(modelfile))
    isai,aiprobability=check_input(
        model,
        vocab,
        inputtext,
        tokenizer=tokenizer,
    )
    return isai


def append_to_csv(filename, data, delimiter):
    # Check if the file exists
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(["Classification", "Text"])
        
        # Append the data
        writer.writerow(data)


def finetuneModel(inputmodel,inputdataset,outputmodelfile,outputvocabfile,percentsplit=0.2,classificationlabel="classification",textlabel="text",epochs=100,lowerbound=0.3, upperbound=0.5):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    traintxt, test_text, train_labels, test_labels = load_data(inputdataset,percentsplit=percentsplit,classificationlabel=classificationlabel,textlabel=textlabel)
    vocab, trainseqs, testseqs = tokenize_data(
        traintxt,
        test_text,
        tokenizer
        
    )
    model = AiDetector(len(vocab))
    
    model.load_state_dict(torch.load(inputmodel))
    # Pass a sample input through the model to compute the size of the convolutional layer output
    _ = model(torch.zeros(1, trainseqs.size(1)).long())
    model.add_fc_layer()
    spinner = Halo(text='finetuning model', spinner='dots')
    spinner.start()

    train_model(model,
                trainseqs,
                torch.tensor(train_labels.values, dtype=torch.float),
                testseqs,
                torch.tensor(test_labels.values, dtype=torch.float),
                epochs=epochs,
                lowerbound=lowerbound,
                upperbound=upperbound
    )
    spinner.stop()

    #print("training complete. Saving...")
    
    save_model(
        model, outputmodelfile
    )

    # Save the vocabulary
    save_vocab(vocab=vocab, vocaboutputfile=outputvocabfile)


def list_files_by_age(directory, extension):
    # Get a list of all files with the given extension
    files = glob.glob(f"{directory}/*.{extension}")
    
    # Sort the files by their last modified time (oldest first)
    files.sort(key=os.path.getmtime)
    
    return files


def evalModel(testfile,vocabfile,modelfile):
    try:
        tokenizer=get_tokenizer()
    except Exception as e:
        if "Can't find model" in str(e):
            tokenizer=get_tokenizer(download=True)
    vocab=load_vocab(vocabfile)
    model = AiDetector(len(vocab))
    df = pd.read_csv(
        testfile,
        sep=detect_delimiter(testfile,"classification","text")
    )
    accuracysum=0
    accuracycount=0
    for index,row in df.iterrows():
        classification=row['classification']
        text=str(row['text'])
        if len(text)>0:
            try:
                model.load_state_dict(torch.load(modelfile))
                isai,aiprobability=check_input(
                    model,
                    vocab,
                    text,
                    tokenizer=tokenizer,
                )
                if isai == classification:
                    accuracysum +=1
                accuracycount+=1
            except Exception as e:
                print(text)
                print(e)
    evaluationpercent=accuracysum/accuracycount
    return evaluationpercent


#instead of reinventing the wheel I'm just going to have a current and fullcsv. It's not really any overhead.

#We need to evaluate the model, and only keep it if it's performing better.
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
    currentmodel=return_specific_files(args.currentdir,".aidetectormodel")[0]
    currentvocab=return_specific_files(args.currentdir,".vocab")[0]
    try:
        currentcsv=return_specific_files(args.currentdir,".csv")[0]
    except:
        currentcsv=os.path.join(args.currentdir,str(uuid.uuid4()).replace("-","")+".csv")

    bookcache=createCache()
    # Get the current working directory
    current_directory = os.getcwd()

    # Create a new directory path
    olddir = os.path.join(current_directory, args.olddir)
    fullldataset=os.path.join(olddir,args.fulldataset)
    evaldataset=None
    if args.evaldataset is None:
        evaldataset=os.path.join(args.currentdir,str(uuid.uuid4()).replace("-","")+".csv")
        #we need to create it
        cmd=['python3','build_dataset.py',"--filename",evaldataset,"--aigen","--modeldir",args.modeldir]
        subprocess.run(cmd)
    else:
        evaldataset=args.evaldataset
    #We're going to need to evaluate the model first...
    currentevalpercent=evalModel(evaldataset,vocabfile=currentvocab,modelfile=currentmodel)
    
    # Create the new directory
    os.makedirs(olddir, exist_ok=True)
    currentcounter=0
    with Halo(text="Generating dataset...", spinner=random.choice(spinner_styles)) as spinner:
        while True:
            rawtext,text=getRandomBookText(bookcache)
            sentences=getSentences(text)
            sentgroups=group_sentences(sentences)
            for sent in sentgroups:
                spinner.text="Iterating through the phrases in the book..."
                spinner.spinner=random.choice(spinner_styles)
                isai=testPhrase(sent,currentmodel,currentvocab)
                if isai==1:
                    spinner.text="Got a human made phrase the model thought was AI!"
                    spinner.spinner=random.choice(spinner_styles)
                    data=[0,sent]
                    append_to_csv(currentcsv,data,delimiter=DELIMITER)
                    append_to_csv(fullldataset,data,delimiter=DELIMITER)
                    currentcounter+=1
                currentllmlist=list_files(args.modeldir)
                llm=None
                while llm==None:
                    spinner.text="Selecting an LLM to generate the text with..."
                    spinner.spinner=random.choice(spinner_styles)
                    #just in case we run into incompatible llms...
                    modelpath=random.choice(currentllmlist)
                    try:
                        llm=Llama(model_path=modelpath,verbose=False,)
                    except Exception as e:
                        if "really a GGML file" in str(e):
                            llm=None
                            os.remove(modelpath)
                outputtext=generateAIOutput(sent,llm)
                while testPhrase(outputtext,currentmodel,currentvocab) == 0:
                    spinner.text="Fooled the classifier model! Generated an AI output which looks 'human' enough..."
                    spinner.spinner=random.choice(spinner_styles)
                    data=[1,outputtext]
                    append_to_csv(currentcsv,data,delimiter=DELIMITER)
                    append_to_csv(fullldataset,data,delimiter=DELIMITER)
                    currentcounter+=1
                    outputtext=generateAIOutput(sent,llm,inputprompt="Q: This text was determined to be AI Generated: "+outputtext+" Rewrite it to not be detectable as AI. A:")
            if currentcounter>=args.traincounter:
                spinner.text="Looks like we have enough data now, finetuning..."
                spinner.spinner=random.choice(spinner_styles)
                #now we need to finetune the new model...
                newname=str(uuid.uuid4()).replace("-","")
                newmodel=os.path.join(args.currentdir,newname+".aidetectormodel")
                newvocab=os.path.join(args.currentdir,newname+".vocab")
                finetuneModel(inputmodel=currentmodel,inputdataset=currentcsv,outputmodelfile=newmodel,outputvocabfile=newvocab)
                #now we only want to keep this thing if it's performing better than the current model, naturally
                newevalpercent=evalModel(evaldataset,vocabfile=newvocab,modelfile=newmodel)
                if newevalpercent >= currentevalpercent:
                    spinner.text="The new classifier performed better!"
                    spinner.spinner=random.choice(spinner_styles)
                    currentevalpercent=newevalpercent
                    #do we want to generate a new dataset to evaluate against? If it was randomly generated, is it truly going to matter? How will we really track progression?
                    #how about a threshold of "its too good against this dataset"
                    
                    move_file(currentmodel,olddir)
                    move_file(currentvocab,olddir) #logically their names should be the same, might not be perfect but whatever it's an archive.
                    oldmodellist=list_files_by_age(olddir,".aidetectormodel")
                    oldvocablist=list_files_by_age(olddir,".vocab")
                    oldmodellist.reverse()
                    oldvocablist.reverse()
                    while len(oldmodellist) > args.archivecount:
                        os.remove(oldmodellist.pop())
                    while len(oldvocablist) > args.archivecount:
                        os.remove(oldvocablist.pop)
                    os.remove(currentcsv)
                    currentcsv=os.path.join(args.currentdir,str(uuid.uuid4()).replace("-","")+".csv")
                    currentmodel=newmodel
                    currentvocab=newvocab
                    while currentevalpercent > evalthreshold: #if our new model is beating the dataset by 98%? 
                        spinner.text="Our model is performing higher than our threshold. Lets see if we can generate better data to stump it..."
                        spinner.spinner=random.choice(spinner_styles)
                        cmd=['python3','build_dataset.py',"--filename",evaldataset,"--aigen","--modeldir",args.modeldir]
                        subprocess.run(cmd)
                        currentevalpercent=evalModel(evaldataset,vocabfile=currentvocab,modelfile=currentmodel)
                else:
                    spinner.text="This new model doesn't perform better than our current model. Let's trash it and try again."
                    spinner.spinner=random.choice(spinner_styles)
                    os.remove(newmodel)
                    os.remove(newvocab)
                currentcounter=0 #give it more data and train, either on a existing dataset or a brand new one.
        



if __name__ == "__main__":
    main()
