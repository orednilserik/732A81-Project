from topic_modelling import *

def analyze_clusters(df_papers, kmeans, topics, embeddings):
    # add cluster assignments to the dataframe
    df_papers['cluster'] = labels

    # extract year from published date
    df_papers['year'] = df_papers['published'].str[:4].astype(int)

    # initialize plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # calculate basic stats for the clusters
    cluster_stats = []

    for cluster_id in sorted(df_papers['cluster'].unique()):
        cluster_papers = df_papers[df_papers['cluster'] == cluster_id]

        stats = {
            'cluster': cluster_id,
            'paper_count': len(cluster_papers),
            'avg_year': cluster_papers['year'].mean(),
            'year_range': (cluster_papers['year'].min(),
                         cluster_papers['year'].max())}

        # get 5 top key terms from topics
        stats['key_terms'] = [term for term, _ in topics[cluster_id][:5]]

        # find representative paper for each cluster i.e. closest to center
        cluster_embeddings = embeddings[df_papers['cluster'] == cluster_id]
        center_dist = np.linalg.norm(cluster_embeddings - kmeans.cluster_centers_[cluster_id], axis=1)
        stats['representative_paper'] = cluster_papers.iloc[center_dist.argmin()]['title']

        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)

    # visualizations
    # cluster size
    sns.barplot(
        data=stats_df,
        x='cluster',
        y='paper_count',
        ax=axes[0,0]
    )
    axes[0,0].set_title('Papers per Cluster')

    # year distributions
    sns.boxplot(
        data=df_papers,
        x='cluster',
        y='year',
        ax=axes[0,1]
    )
    axes[0,1].set_title('Year Distribution by Cluster')

    # topic similarities
    similarity_matrix = cosine_similarity(kmeans.cluster_centers_)
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.2f',
        ax=axes[1,0]
    )
    axes[1,0].set_title('Topic Similarity')

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    sns.scatterplot(
        x=embeddings_2d[:,0],
        y=embeddings_2d[:,1],
        hue=df_papers['cluster'],
        ax=axes[1,1]
    )
    axes[1,1].set_title('Cluster Visualization (t-SNE)')

    plt.tight_layout()

    # cluster summaries
    print("\nCluster Summaries:")
    for stats in cluster_stats:
        print(f"\nCluster {stats['cluster']}:")
        print(f"Size: {stats['paper_count']} papers")
        print(f"Time span: {stats['year_range'][0]} - {stats['year_range'][1]}")
        print(f"Key terms: {', '.join(stats['key_terms'])}")
        print(f"Representative paper: {stats['representative_paper']}")

    return stats_df

stats_df = analyze_clusters(df_papers, kmeans, topics, embeddings)