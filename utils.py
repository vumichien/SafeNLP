""" 
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
Utility fuctions 
"""

import argparse
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to evaluation dataset. i.e. implicitHate.json or toxiGen.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True,
                        help="a local path to a model or a model tag on HuggignFace hub.")
    parser.add_argument('--lmHead', type=str, required=True,
                        choices=['mlm', 'clm'])
    parser.add_argument('--config', type=str,
                        help='Path to model config file')
    parser.add_argument("--force", action="store_true", 
                        help="Overwrite output path if it already exists.")
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args, from_tf=False):
    '''
    Load tokenizer and model to evaluate.
    '''
    # load Causal Language Model Head
    pretrained_weights = args.model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    model = AutoModelForCausalLM.from_pretrained(pretrained_weights, quantization_config=bnb_config,
                      low_cpu_mem_usage=True, device_map={"":0})

    model = model.eval()

    return tokenizer, model
