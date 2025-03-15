from feature_extract import *

def model_topics(embeddings, n_topics):


    # fit k-means on SciBERT embeddings
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # get top terms for each cluster
    topics = []
    for i in range(n_topics):
        # get indices of documents in this cluster
        cluster_docs = df_papers['processed_text'][cluster_labels == i]

        # get 10 most common terms in cluster
        cluster_text = ' '.join(cluster_docs)
        words = cluster_text.split()
        word_freq = Counter(words).most_common(10)

        topics.append(word_freq)

    return kmeans, topics, cluster_labels

# find optimal number of topics using silhouette score
silhouette_scores = []
K = range(3, 15)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    silhouette_scores.append(score)

optimal_topics = K[np.argmax(silhouette_scores)]

# create and print topics
kmeans, topics, labels = model_topics(embeddings, n_topics=optimal_topics)

for idx, topic_words in enumerate(topics):
    print(f"\nTopic {idx + 1}:")
    print(", ".join([f"{word}({count})" for word, count in topic_words]))
    
# plot the silhouette scores
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Number of Topics (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Topics')
plt.grid(True)
plt.show()