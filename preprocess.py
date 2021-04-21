def load_data(filename):
    """
    Pre-process CONLL dataset for NER model training with spaCy. 
    
    Splits data into sentences, stored as lists of tokens:
        
        `[['Only', 'France', 'and', 'Britain', 'backed', 'Fischler', ''s', 'proposal', '.']]`
    
    Maps "PER" tag to "PERSON" and ignores other tags ("LOC", "ORG"). For simplicity 
    working with spaCy, relevant entities are stored as lists of (`start`, `end`, `tag`) tuples:

        `[[(31, 39, 'Fischler')]]`  

    Args:
        filename (str): path to dataset. 
    Returns:
        Tuple of sentence tokens (list of lists) and entity tags (list of lists).
    """
    with open(filename, "r") as f:
        
        # Initialize empty lists, sublists, and char count
        sentences, tokens = [], []
        entities, tags = [], []
        idx = 0
        
        # Parse sentence tokens and named entity tags
        for line in f:
            if line != "\n":  # sentence starts
                token, _, _, tag = line.split()
                if token != "-DOCSTART-":
                    tokens.append(token)
                    if tag == "I-PER":
                        tags.append((idx, idx + len(token), "PERSON"))
                    idx += len(token) + 1
                    
            elif line == "\n" and len(tokens) > 0:  # sentence ends
            
                # Append current sentence's tokens and tags to lists
                sentences.append(tokens)
                entities.append(tags)
                
                # Reset sublists and char count
                tokens = []
                tags = []
                idx = 0
    
    return sentences, entities
