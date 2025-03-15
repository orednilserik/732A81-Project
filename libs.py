#!/home/wountain/miniconda3/envs/textmining/bin/python
# for fetching data
import urllib.parse
import requests
import xmltodict
import pandas as pd
import time
from tqdm import tqdm

# for preprocessing
import spacy
import scispacy

# for embeddings
from sentence_transformers import SentenceTransformer

# for topic modelling/clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from collections import Counter

# for cluster analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity