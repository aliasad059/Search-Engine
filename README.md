# Simple Search Engine with Information Retrieval Techniques

This Python project implements a basic search engine with information retrieval techniques. It provides functionalities for searching and retrieving relevant documents from a dataset. Here's a brief overview of the project components:

## Features
- **Data Loading and Preprocessing:** The code can load a dataset from a JSON file and preprocess it, including normalization, tokenization, removing stopwords, punctuation, and stemming.

- **Indexing:** The project creates an inverted index of the dataset. This index stores information about the occurrence of words in the documents, including word frequency, document frequency, and positions within documents. It also includes a champion list to optimize searches.

- **Searching:** The search engine supports various types of queries, including single-word queries, phrasal queries (queries enclosed in double quotes), and negation queries (queries with '!' for exclusion). It ranks search results based on relevance to the query using TF-IDF scores.

- **Ranking and Retrieval:** The code provides ranked retrieval functionality to retrieve the top-k most relevant documents to a query using TF-IDF scores. It can optionally utilize champion lists to speed up ranked retrieval.

- **Zipf's Law Visualization:** The project includes a function to draw a Zipf's Law plot, showing the distribution of word frequencies in the dataset.

- **Index Saving and Loading:** The code allows for saving and loading the created index to/from a file, improving efficiency when working with large datasets.

## Implementation Details
This project implements a search engine from scratch with the following features:
- Data preprocessing, including tokenization, normalization, and stemming.
- Creation of an inverted index for efficient word-document mapping.
- Support for various query types, including phrasal and negation queries.
- Ranked retrieval search using TF-IDF scores.
- Optimization of query answering speed using a champion list.

## Getting Started
To use this search engine, you need:
- Python 3.6 or higher
- Required Python libraries: `json`, `itertools`, `collections`, `matplotlib`, `hazm`, `numpy`, `pandas`, `re`, `string`

## Usage
- Load and preprocess your dataset.
- Create an inverted index and a champion list for your dataset.
- Use the search engine to perform various types of searches, including ranked retrieval.
- Display search results to the user.

## Example Usage
Here's an example of how to use this search engine:

```python
# Load and preprocess your dataset
documents = [...]  # Your dataset
inverted_index = create_inverted_index(documents)
champion_list = create_champion_list(inverted_index, k=10)

# Perform a search
query = "your user query"
results = ranked_retrieval(query, inverted_index, champion_list, k=10)
display_results(results)
```

This Python project implements a basic search engine with information retrieval techniques. It provides functionalities for searching and retrieving relevant documents from a dataset. For more advanced features, accuracy, and scalability, you can also explore another repository that uses the Elasticsearch framework for similarity modeling, spell correction, and clustering.

## Additional Features in the Advanced Repository
- **Elasticsearch Integration:** The advanced repository integrates Elasticsearch, a powerful search and analytics engine, to enhance search capabilities.
- **Similarity Modeling:** Elasticsearch allows for more advanced similarity modeling, enabling better relevance ranking of search results.
- **Spell Correction:** The repository incorporates spell correction mechanisms to improve search accuracy, ensuring that users get relevant results even with typos or misspelled queries.
- **Clustering:** It includes clustering algorithms to group and retrieve similar documents, providing users with a more structured and organized search experience.

You can find the advanced repository using the following link:
[Search Engine with Elasticsearch](https://github.com/aliasad059/Elastic-Search)

Please refer to the advanced repository for more sophisticated search capabilities and accuracy enhancements.
