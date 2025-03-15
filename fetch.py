from libs import *

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

        for i in tqdm(range(num_iterations), desc=f"Fetching {q}"):
            # full url of API fetch
            url = f"{base_url}search_query={search_query}&start={0}&max_results={1000}&sortBy=submittedDate&sortOrder=descending"

            try:
                response = requests.get(url) # fetch
                response.raise_for_status() # raise HTTPerror if fetch doesnt work
                data = xmltodict.parse(response.text) # parse fetched data to a dictionary

                entries = data['feed'].get('entry', []) # get list of papers
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
            except Exception:
                print(f"Error fetching ArXiv papers for query '{query}'")
                continue

    # convert to DataFrame and remove duplicates
    df = pd.DataFrame(papers)
    df = df.drop_duplicates(subset=['title'])

    return df


# queries based on keyword search
queries = [
    '(intermediate fusion OR early fusion OR late fusion OR hybrid fusion OR feature fusion) AND (medical imaging OR healthcare)',
    '(multimodal OR cross-modal OR multi-stream) AND (MRI OR CT OR X-Ray OR PET OR Ultrasound OR DICOM)',
    '(transformer OR self-attention OR cross-attention OR multi-head attention) AND (medical imaging OR healthcare)',
    '(vision-language OR multi-modal) AND (medical diagnosis OR clinical)',
    '(deep learning OR neural network) AND (multimodal fusion OR cross-modal fusion) AND medical'
]

df_papers = arxiv_fetch(queries)

print(f"\nTotal papers collected: {len(df_papers)}")

df_papers.head()