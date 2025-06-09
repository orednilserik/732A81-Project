# %%
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
from sentence_transformers import SentenceTransformer, models

# for topic modelling/clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

# for cluster analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# %%
def arxiv_fetch(query, max_res_per_query = 1000, batch_size = 100, num_iterations = 10):
    """
    queries: List of search queries
    max_results_per_query: maximum results to fetch per query
    batch_size: number of results per API call (max 100 according to ArXiv rules)

    Returns:
        pd.DataFrame: DataFrame containing all unique paper data
    """
    base_url = 'http://export.arxiv.org/api/query?'
    papers = []

    for q in query:
        search_query = urllib.parse.quote(q) # parse the search word query

        total_iterations = min(num_iterations, max_res_per_query // batch_size) # calculate total iterations

        for i in tqdm(range(total_iterations), desc=f"Fetching {q}"):
            # full url of API fetch
            start_index = i * batch_size
            url = f"{base_url}search_query={search_query}&start={start_index}&max_results={batch_size}&sortBy=submittedDate&sortOrder=descending"

            try:
                response = requests.get(url, timeout=30) # fetch
                response.raise_for_status() # raise HTTPerror if fetch doesnt work
                data = xmltodict.parse(response.text) # parse fetched data to a dictionary

                entries = data['feed'].get('entry', []) # get list of papers
                if not entries:
                    print(f"No more results for query: {q}")
                    break # if no entries, break the loop

                entries = [entries] if not isinstance(entries, list) else entries # ensures a single paper fetch is a list as well

                # extract the columns of interest and append
                for entry in entries:
                    paper = {
                        'title': entry['title'].replace('\n', ' ').strip(),
                        'abstract': entry['summary'].replace('\n', ' ').strip(),
                        'published': entry['published'][:10],
                    }

                    papers.append(paper)

                # ArXiv API rate limit: 3 second delay between requests
                time.sleep(3)
            # error handling
            except Exception as e:
                print(f"Error fetching ArXiv papers for query '{q}': {e}")
                continue

    # convert to DataFrame and remove duplicates
    df = pd.DataFrame(papers)
    df = df.drop_duplicates(subset=['title'])

    return df


# queries based on keyword search
queries = [
    '(early fusion OR late fusion OR intermediate fusion OR decision fusion OR feature fusion OR modality fusion) AND (deep learning OR neural network)',
    '(cross-modal attention OR multimodal transformer OR vision-language model OR CLIP OR ViLT OR LXMERT) AND (fusion OR multimodal)',
    '(multimodal OR multi-modal OR cross-modal) AND (medical imaging OR radiology OR pathology OR MRI OR CT OR X-ray OR ultrasound OR PET)',
    '(multimodal fusion OR cross-modal fusion) AND (clinical diagnosis OR medical diagnosis OR healthcare OR biomedical)',
    '(graph neural network OR GNN OR attention mechanism OR self-attention OR cross-attention) AND (multimodal OR multi-modal) AND medical',
    '(vision-language OR text-image OR image-text) AND (medical report OR radiology report OR clinical notes OR pathology)',
    '(contrastive learning OR self-supervised learning OR few-shot learning) AND (multimodal OR cross-modal) AND (medical OR healthcare)',
    '(multi-stream OR ensemble OR hybrid model) AND (medical imaging OR healthcare) AND (fusion OR multimodal)'
]

df_papers = arxiv_fetch(queries)

print(f"\nTotal papers collected: {len(df_papers)}")

df_papers.head()

# %%
nlp = spacy.load("en_core_sci_sm") # scispacy for biomedical preprocessing of data


def preprocess_text(text):

    if pd.isna(text) or not text.strip():
        return ''

    doc = nlp(text.lower())
    words = []

    for token in doc:
      if token.ent_type_: # named entities in the corpus are always kept
            words.append(token.text)
      # apply filtering for other tokens
      elif (not token.is_punct and
              not token.is_space and
              not token.like_url and
              not token.like_email and
              len(token.text) > 1 and
              token.text.isalpha()):
            # keep token even if its a stopword for domain-specific analysis
            words.append(token.lemma_ if token.lemma_ != '-PRON-' else token.text)

    return ' '.join(words)

# remove papers with missing abstracts
df_papers = df_papers.dropna(subset=['abstract'])
df_papers = df_papers[df_papers['abstract'].str.strip() != '']

# apply preprocessing on abstract
df_papers['processed_text'] = df_papers['abstract'].apply(preprocess_text)

# remove papers where preprocessing resulted in empty text
df_papers = df_papers[df_papers['processed_text'].str.strip() != '']

# final deduplication based on processed text similarity
df_papers = df_papers.drop_duplicates(subset=['processed_text'])

print(f"Final dataset size after preprocessing: {len(df_papers)}")

# %%
# initialize SciBERT model which is optimized for scientific text
# https://huggingface.co/allenai/scibert_scivocab_uncased
#model = SentenceTransformer('allenai/scibert_scivocab_uncased')

word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def create_embeddings(df):
    all_embs = []

    for text in tqdm(df['processed_text'].tolist(), desc="Creating embeddings"):
        # treat heavily preprocessed text as single document since sentence boundaries might be lost
        if len(text.strip()) == 0:
            # fallback for empty text
            doc_emb = np.zeros(model.get_sentence_embedding_dimension())
        else:
            # encode the entire processed text as one unit
            doc_emb = model.encode(text)

        all_embs.append(doc_emb)
    return np.array(all_embs)

# generate embeddings
embeddings = create_embeddings(df_papers)

# %%
# some specific stopwords
domain_stopwords = [
    'study', 'method', 'result', 'analysis', 'approach', 'data', 'using', 'based',
    'model', 'patient', 'patients', 'algorithm', 'performance', 'demonstrate',
    'learning', 'training', 'dataset', 'clinical', 'research', 'significantly',
    'use', 'models', 'image', 'images', 'methods', 'propose', 'framework',
    'provide', 'include', 'introduce', 'high', 'network',
    'https', 'github', 'com', 'available', 'www', 'http', 'org',
    'code', 'github com', 'https github', 'https github com', 'available https',
    'available https github'
]

k_range=(3,10)

def find_optimal_clusters(embeddings, k_range=k_range):
    """find optimal number of clusters using silhouette score"""

    scores = []
    K = range(k_range[0], k_range[1])

    for k in tqdm(K, desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # only calculate score if we have valid clusters
        if len(np.unique(labels)) == k:
            score = silhouette_score(embeddings, labels)
        else:
            score = -1

        scores.append(score)

    # find best k (excluding invalid scores)
    valid_scores = [(i, score) for i, score in enumerate(scores) if score != -1]
    if not valid_scores:
        return k_range[0] # fallback if no valid scores

    best_idx = max(valid_scores, key=lambda x: x[1])[0]
    return K[best_idx]

def extract_topic_terms(docs, labels, n_clusters, top_n=8):
    """extract representative terms for each cluster using TF-IDF"""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1500,
        min_df=3,
        max_df=0.6,
        ngram_range=(1, 3)
    )

    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    topics = {}
    for cluster_id in range(n_clusters):
        # get documents in this cluster
        cluster_mask = labels == cluster_id
        cluster_docs = np.where(cluster_mask)[0]

        if len(cluster_docs) == 0:
            topics[cluster_id] = []
            continue

        # calculate mean TF-IDF scores for this cluster
        cluster_tfidf = tfidf_matrix[cluster_docs].mean(axis=0).A1

        # get top terms, filtering domain stopwords
        top_indices = cluster_tfidf.argsort()[::-1]

        terms = []
        for idx in top_indices:
            term = feature_names[idx]
            score = cluster_tfidf[idx]

            # filter out domain stopwords and low scores
            if term not in domain_stopwords and score > 0.01:
                terms.append((term, score))

            if len(terms) >= top_n:
                break

        topics[cluster_id] = terms

    return topics

def analyze_cluster_quality(embeddings, labels):
    """analyze clustering quality"""
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(embeddings, labels)

    # cluster size distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette score: {silhouette_avg:.3f}")
    print(f"Cluster sizes: {cluster_sizes}")

    return silhouette_avg, cluster_sizes


# PCA
n_components = min(100, embeddings.shape[1], embeddings.shape[0]-1)
pca = PCA(n_components=n_components, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

# clustering
optimal_k = find_optimal_clusters(embeddings_pca)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings_pca)

# silhouette
silhouette_avg, cluster_sizes = analyze_cluster_quality(embeddings_pca, labels)

# topic extraction
topics = extract_topic_terms(df_papers['processed_text'], labels, optimal_k)

# output
print(f"\n{'='*50}")
print("Discovered topics")
print(f"{'='*50}")
for cluster_id, terms in topics.items():
    cluster_size = cluster_sizes.get(cluster_id, 0)
    print(f"\nTopic {cluster_id + 1} ({cluster_size} papers):")
    if terms:
        print("  " + ", ".join([f"{term}({score:.3f})" for term, score in terms]))
    else:
        print("No significant terms found")

df_papers_clustered = df_papers.copy()
df_papers_clustered['cluster'] = labels
df_papers_clustered['topic_id'] = labels + 1


# %%
def analyze_clusters(df_papers, kmeans_model, topics, labels, reduced_embeddings):
    """
    analyze clusters
    """
    # working copy
    df_analysis = df_papers.copy()
    df_analysis['cluster'] = labels
    df_analysis['year'] = df_analysis['published'].str[:4].astype(int)

    cluster_summaries = []

    for cluster_id in sorted(df_analysis['cluster'].unique()):
        cluster_papers = df_analysis[df_analysis['cluster'] == cluster_id]

        # basic characteristics
        summary = {
            'cluster_id': cluster_id,
            'topic_label': f"Topic {cluster_id + 1}", # index from 1
            'paper_count': len(cluster_papers),
            'percentage': (len(cluster_papers) / len(df_analysis)) * 100,
        }

        # temporal analysis
        summary.update({
            'time_span': (cluster_papers['year'].min(), cluster_papers['year'].max()),
            'avg_year': cluster_papers['year'].mean(),
            'recent_papers': len(cluster_papers[cluster_papers['year'] >= 2020])
        })

        # topic terms
        if cluster_id in topics and topics[cluster_id]:
            summary['key_terms'] = [term for term, _ in topics[cluster_id][:5]]
            summary['term_scores'] = topics[cluster_id][:5]  # scores for reference
        else:
            summary['key_terms'] = []
            summary['term_scores'] = []

        # representative papers
        cluster_mask = df_analysis['cluster'] == cluster_id
        cluster_embeddings = reduced_embeddings[cluster_mask]

        if len(cluster_embeddings) > 0:
            center = kmeans_model.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)

            # 3 papers
            closest_indices = np.argsort(distances)[:3]
            cluster_papers_subset = cluster_papers.reset_index(drop=True)

            summary['representative_papers'] = []
            for idx in closest_indices:
                if idx < len(cluster_papers_subset):
                    paper = cluster_papers_subset.iloc[idx]
                summary['representative_papers'].append({
                    'title': paper['title'],
                    'year': paper['year'],
                    'distance_to_center': distances[idx]
                })

        cluster_summaries.append(summary)

    return pd.DataFrame(cluster_summaries)

def print_cluster_summary(cluster_df):
    """clean summary of clusters"""
    print("=" * 80)
    print("CLUSTER ANALYSIS SUMMARY")
    print("=" * 80)

    total_papers = cluster_df['paper_count'].sum()
    print(f"Total papers analyzed: {total_papers}")
    print(f"Number of clusters: {len(cluster_df)}")
    print()
    # print each cluster summary
    for _, cluster in cluster_df.sort_values('paper_count', ascending=False).iterrows():
        print(f"{cluster['topic_label']} ({cluster['paper_count']} papers, {cluster['percentage']:.1f}%)")
        print(f"Time span: {cluster['time_span'][0]}-{cluster['time_span'][1]} (avg: {cluster['avg_year']:.0f})")

        if cluster['key_terms']:
            terms = ", ".join(cluster['key_terms'])
            print(f"   Key terms: {terms}")

        if cluster['representative_papers']:
            print("Representative papers:")
            for i, paper in enumerate(cluster['representative_papers'], 1):
                print(f"{i}. {paper['title']} ({paper['year']})")

        print()

cluster_analysis = analyze_clusters(df_papers, kmeans, topics, labels, embeddings_pca)
print_cluster_summary(cluster_analysis)


# %%
def visualize_clusters_individual(df_papers, kmeans, topics, embeddings, reduced_embeddings):
    
    # prep data
    df_with_clusters = df_papers.copy()
    df_with_clusters['cluster'] = kmeans.labels_
    df_with_clusters['year'] = df_with_clusters['published'].str[:4].astype(int)
    unique_clusters = sorted(df_with_clusters['cluster'].unique())

    # cluster sizes with percentages
    plt.figure(figsize=(10, 6))
    cluster_counts = cluster_analysis.sort_values('paper_count', ascending=False)
    display_ids = [str(int(cid) + 1) for cid in cluster_counts['cluster_id']]
    bars = plt.bar(display_ids, cluster_counts['paper_count'])
    plt.title('Papers per Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Papers')

    # percentage labels on bars
    for bar, percentage in zip(bars, cluster_counts['percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    # temporal distribution as box plots
    plt.figure(figsize=(6, 6))
    cluster_years = []
    cluster_labels = []

    for cluster_id in unique_clusters:
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]['year']
        cluster_years.append(cluster_data.values)
        cluster_labels.append(str(cluster_id + 1))

    box_plot = plt.boxplot(cluster_years, tick_labels=cluster_labels, patch_artist=True)
    plt.title('Temporal Distribution by Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster')
    plt.ylabel('Publication Year')
    plt.grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    plt.show()

    # visualization of clusters in embedding space
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],§
                         c=kmeans.labels_, cmap='tab10', alpha=0.6, s=20)
    plt.title('Cluster Visualization', fontsize=14, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # cluster centroids with labels
    for cluster_id in unique_clusters:
        cluster_points = reduced_embeddings[kmeans.labels_ == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            plt.scatter(centroid[0], centroid[1], c='red', s=100, marker='x', linewidths=3)
            plt.text(centroid[0], centroid[1], str(cluster_id + 1), fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    # topic similarity heatmap
    plt.figure(figsize=(6, 6))
    if hasattr(kmeans, 'cluster_centers_'):
        similarity_matrix = cosine_similarity(kmeans.cluster_centers_)
        similarity_matrix = (similarity_matrix+1)/2
        cluster_labels_sim = [str(i+1) for i in range(len(similarity_matrix))]

        sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=cluster_labels_sim, yticklabels=cluster_labels_sim,
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Topic Similarity Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Cluster')
    else:
        plt.text(0.5, 0.5, 'No similarity data available', ha='center', va='center',
                transform=plt.gca().transAxes)
        plt.title('Topic Similarity Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

visualize_clusters_individual(df_papers, kmeans, topics, labels, embeddings_pca)
