# aidetector-example

This project generates a dataset for aidetector based on the works found in Project Gutenberg and an LLM of your choice.

## Installation

1. Clone this repository.
2. Install the required Python packages:

    ```
    pip3 install -r requirements.txt
    ```

## Usage

You can run the script with the following command:

```
python3 build_dataset.py --filename output.csv --books 10 --aigen --model /path/to/your/model
```

Here's what each argument does:

- `--filename`: The name of the file to write. This argument is required.
- `--books`: The number of Gutenberg books to randomly sample from. The default is 10.
- `--aigen`: If this flag is thrown, the script will generate AI output to store in the dataset. You must also specify a model using `--model`.
- `--model`: Specify a model to use with `--aigen`. This argument is required when `--aigen` is used.
- `--quotes`: If this flag is thrown, the script will pull quotes from the internet to provide something to read while AI is generating output.
- `--readbook`: If this flag is thrown, the script will select one of the books at random and provide its story, sentence by sentence, providing something to read while the AI is generating text. If both `--quotes` and `--readbook` are thrown, `--quotes` will be ignored.

The script will generate a CSV file with the specified filename. The CSV file will have two columns: "classification" and "text". The "classification" column will contain 1 for AI-generated text and 0 for human-generated text. The "text" column will contain the text.

Given the nature of AI model architectures, generating AI output for this dataset can be a time-consuming process. However, it's important to note that this script is designed for illustrative purposes and it generates a new, randomized dataset with each run. This provides the flexibility to create diverse datasets sourced from the public domain works available on Project Gutenberg. Moreover, you have the freedom to choose a model compatible with llama.cpp to generate the AI output, allowing you to tailor the process to your specific needs and preferences.
