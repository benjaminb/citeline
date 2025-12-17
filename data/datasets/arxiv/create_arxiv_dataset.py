import csv
import difflib
import json
import logging
import os
import pymupdf
import re
import requests
from datetime import datetime
from dotenv import load_dotenv
from itertools import islice
from semantic_text_splitter import TextSplitter
from time import sleep
from tqdm import tqdm
