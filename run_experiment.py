"""
To run an experiment:

database: 'test'
table: 'test_bge'


specify 
- the database
- the database vector table to use (this sets the embedding model, normalization used or not)
- the distance metric to use, cosine, l2, or ip
- the % of trivial examples relative to nontrivial examples to use in the training set
- the method you'll use to augment each sentence in the training set
    -e.g. previous n sentences, title + abstract, etc.



user specifies a vector table in the database they want to run the experiment on, plus a distance metric

Then we load the training set into memory from dataset/no_reviews/
    - load all nontrivial examples
    - load a portion of trivial examples
    
set up separate lists for trivial and nontrivial examples' IoU scores
For each example in training set, get the `sent_no_cit` value from the example
    - do this in batches
    - use the specified augmentation function to enrich the sent_no_cit
    - embed the enriched sentence
    - get the top k nearest neighbors, particularly their doi's
    - compute intersection over union 
    - log the predicted top doi's for the sentence
        - log trivial and nontrivial examples separately

After all examples are processed, compute the average intersection over union for the training set
    - average IoU for trivial examples
    - average IoU for nontrivial examples
    - average IoU for all examples

Write out results to file


"""
import argparse
import os
import yaml
from dotenv import load_dotenv
from database.database import DatabaseProcessor

def argument_parser():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run an experiment with specified configuration.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to the YAML configuration file.')

    return parser.parse_args()


def main():
    args = argument_parser()
    
    # Load YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    load_dotenv()
    db_params = {
        'dbname': config['database'],
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    processor = DatabaseProcessor(db_params)
    processor.test_connection()


if __name__ == "__main__":
    main()
