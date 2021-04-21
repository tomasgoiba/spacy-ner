## Named entity recognizer
Custom [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) (NER) model with [spaCy](https://spacy.io/). 

### Overview
Python program to train an English NER model for detecting people's names using spaCy.

- [`conll.txt`](./conll.txt) is the raw dataset used to train the model ([CoNLL-2013](https://www.clips.uantwerpen.be/conll2003/ner/))
- [`preprocess.py`](./preprocess.py) contains a helper function to load, parse, and preprocess the raw dataset.
- [`ner.py`](./ner.py) is the main program and does the following:
  - Create the training data from the raw dataset.
  - Train the model according to the specified hyperparameters (`EPOCHS`, `BATCH_SIZE`, `DROPOUT`).
  - Evaluate the model on a subset of the training data whose size is given by `TEST_SIZE`.
  - Save the model to disk (optional).
  - Use the new model or a pre-trained one for inference (optional). 
- [`model`](./model) contains a model trained with the following hyperparameters:
  - Epochs: 100
  - Drop-out rate: 0.1
  - Mini-batch size: 32
- [`harry.txt`](./harry.txt) contains a sample of text with a few named entities to test the model. 

### Instructions
#### Libraries
The following libraries are required to run `ner.py`:
- [spaCy](https://spacy.io/usage)
- [scikit-learn](https://scikit-learn.org/stable/install.html)

#### Arguments
- `--model`: directory containing a pre-trained model (if using a pre-trained model for inference).
- `--data`: path to the raw dataset (if training a fresh model).
- `--save`: directory to save the new model to.
- `--file`: path to a text file containing named entities (if running inference)
- `--output`: output path to save the model predictions (if running inference)

#### Example

- Train a new model and save it:
  <p align="center"><code>python ner.py --data conll.txt --save .</code></p>

- Run inference using a pre-trained model:
  <p align="center"><code>python ner.py --model ./model --file harry.txt --output harry_output.txt</code></p>
  
### Time spent
In total, 3 hours. 
- 1h reviewing spaCy documentation;
- 1.5h coding solution;
- 0.5h debugging.
