"""
Data Scientist (Adarga) - Tomás Gómez
Python Coding Exercise - NER

NER model to detect people's names using spaCy. 

Usage example:

    python3 ner.py --data conll --save . --file text.txt --output text_ner.txt

Time spent on this exercise: 3h
- 1h reviewing spaCy documentation;
- 1.5h coding solution;
- 0.5h debugging.
"""

from preprocess import *

import spacy
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split

import random
import time
import os
import argparse


TEST_SIZE = 0.3
EPOCHS = 30
BATCH_SIZE = 32
DROPOUT = 0.1


def split_data(tokens, entities, test_size=TEST_SIZE):
    """
    Split data into train and test subsets. 

    Args:
        tokens (list of lists): sentences as lists of tokens. 
        entities (list of lists): lists of (`start`, `end`, `tag`) tuples
            for each sentence. 
        test_size (float): proportion of dataset to include in test subset. 
    Returns:
        Tuple of train and test data, as (`tokens`, `entities`) tuples.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        tokens,
        entities,
        test_size=test_size,
        )
    return zip(X_train, y_train), zip(X_test, y_test)


def create_nlp(ents=["PERSON"]):
    """
    Create NLP pipeline with brand new NER, and define its tag scheme. 

    Args:
        ents (list of str): list of entity tags. 
    Returns:
        NLP pipeline, as `Language` object. 
    """
    
    # Create pipeline with NER component
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    
    # Define tag scheme for NER
    for ent in ents:
        ner.add_label(ent)
    return nlp


def get_examples(data, nlp):
    """
    Create `Example` objects from reference data and pipeline prediction. 
    Required to train and evaluate spaCy models. 

    Args:
        data (list of tuples): train/test data. 
        nlp (Language): NLP pipeline. 
    Returns:
        List of `Example` objects. 
    """
    return [Example.from_dict(
        nlp(" ".join(tokens)),  # prediction
        {"entities" : ents}     # reference
        ) for tokens, ents in data]
 

def train_ner(data, nlp, epochs=EPOCHS, batch_size=BATCH_SIZE, dropout=DROPOUT):
    """
    Train NER component of NLP pipeline. 

    Args:
        data (list): list of `Example` objects. 
        nlp (Language): NLP pipeline.
        epochs (int): number of training epochs.
        batch_size (int): number of samples per training batch.
        dropout (float): drop-out rate.
    Returns:
        Updated NLP pipeline with trained NER. 
    """
    for i in range(epochs):
        
        # Initialize empty dictionary to store loss
        losses = {}
        
        # Shuffle data and split into batches
        random.shuffle(data)
        batches = minibatch(data, size=batch_size)
        
        # Mini-batch gradient descent
        for batch in batches:
            nlp.update(
                batch,
                drop=dropout,
                sgd=optimizer,
                losses=losses
            )
        print(f"Epoch {i + 1}: loss of {losses['ner']:.2f}")
    
    return nlp


def get_predictions(infile, outfile, nlp):
    """
    Use NER to predict new entities from a text file.

    Text file must have a single word per line, and sentences
    can optionally be separated by an extra newline.

    Args:
        infile (str): path to input file. 
        outfile (str): path to output NER predictions.
        nlp (Language): NLP pipeline.
    """
    
    # Load input file
    with open(infile, "r") as f:
        tokens = [line.rstrip() for line in f if line != "\n"]

    # Generate predictions
    doc = nlp(" ".join(tokens))
    predictions = [(token.text, token.ent_type_) for token in doc]

    # Write predictions to output file
    with open(outfile, "w") as f:
        for token, ent in predictions:
            ent = ent if ent != "" else "O"
            f.write(f"{token}\t{ent}\n")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a NER model with spaCy."
        )
    parser.add_argument("--data", "-d", type=str, help="Path to CoNLL dataset.")
    parser.add_argument("--model", "-m", type=str, help="Load model from specified directory.")
    parser.add_argument("--save", "-s", type=str, help="Save new model to specified directory.")
    parser.add_argument("--file", "-f", type=str, help="Generate predictions from specified batch file.")
    parser.add_argument("--output", "-o", type=str, help="Desired output path for predictions.")

    args = parser.parse_args()
    if (args.model and args.data) or (not args.model and not args.data):
        parser.error("Use --data to train a new model, or --model to load an existing one.")
    if args.file and not args.output:
        parser.error("Use --output to specify an output path for the model predictions.")

    if not args.model:

        # Load dataset
        print("Loading CoNLL dataset...")
        sentences, entities = load_data(args.data)

        # Split data into train/test subsets
        TRAIN_DATA, test_data = split_data(sentences, entities)

        # Initialize NER and gradient descent optimizer
        nlp = create_nlp()
        optimizer = nlp.initialize()

        # Convert training data to training instances
        print("Creating training data...") 
        train_examples = get_examples(TRAIN_DATA, nlp)

        # Train NER on training instances
        print(f"Training NER on {len(train_examples)} samples...")
        start = time.time()
        nlp = train_ner(train_examples, nlp)
        end = time.time()
        print(f"Training finished in {end - start:.2f} seconds.\n")

        # Evaluate NER on test subset
        test_examples = get_examples(test_data, nlp)
        print(f"Evaluating NER on {len(test_examples)} samples...")
        scores = nlp.evaluate(test_examples)
        print(f"Precision: {scores['ents_p']:.2f}\n"
              f"Recall: {scores['ents_r']:.2f}\n"
              f"F1: {scores['ents_f']:.2f}\n")
        
        # Save model to disk
        if args.save:
            print("Saving model to disk...")
            nlp.to_disk(os.path.join(args.save, "model"))

    else:

        # Load pre-trained model
        print("Loading existing model...")
        nlp = spacy.load(args.model)

    if args.file:

        # Write model predictions to output file
        print("Generating predictions for input file...")
        get_predictions(args.file, args.output, nlp)


