import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import ward, dendrogram
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix


import matplotlib.pyplot as plt


def preprocess_text(text):
    """Preprocess the given text by removing stop words, punctuation and other irrelevant information."""
    # Remove newlines, tabs and consecutive spaces
    text = re.sub(r'\n|\t|\s\s+', ' ', text)
    # Remove date
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    # Remove sender
    text = re.sub(r'From:.*', '', text)
    # Remove path
    text = re.sub(r'Reply-To:.*|Path:.*', '', text)
    # Remove organization
    text = re.sub(r'Organization:.*', '', text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text


def load_data():
    """Load and preprocess the 20 newsgroup dataset."""
    data_dir = '20_newsgroups'
    file_contents = defaultdict(str)
    file_labels = {}
    for label_id, label_name in enumerate(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label_name)
        #print(label_dir)
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, encoding='latin1') as file:
                file_contents[filepath] = preprocess_text(file.read())
            file_labels[filepath] = label_id
    return list(file_contents.values()), list(file_labels.values())



def cluster_documents(data, labels, n_clusters_list, method):
    for n_clusters in n_clusters_list:

        """Cluster the given documents using the specified clustering method and number of clusters."""
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data)
        X = Normalizer().fit_transform(X)

        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
            model.fit(X)
        elif method == 'whc':
            '''''''''
            linkage_matrix = ward(X.toarray())
            dendrogram(linkage_matrix)
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            '''''''''''
            X_dense = X.toarray()
            linkage_matrix = ward(X_dense)
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', compute_full_tree=True)
            print("before fit")
            model.fit(X_dense)
            print("after fit")
            #dendrogram(linkage_matrix)
            #plt.show()
        

            print('Printing clustering progress:')
            children = model.children_


        elif method == 'ac':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            model.fit(X.toarray())

        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            model.fit(X)
        else:
            raise ValueError('Invalid clustering method')
        #model.fit(X)
        #model.fit(X.toarray())
        y_pred = model.labels_
        ami = adjusted_mutual_info_score(labels, y_pred)
        ars = adjusted_rand_score(labels, y_pred)
        comp_score = completeness_score(labels, y_pred)
        print(f'Clustering performance (n_clusters={n_clusters}, method={method}):')
        print(f'Adjusted Mutual Information: {ami:.3f}')
        print(f'Adjusted Random Score: {ars:.3f}')
        print(f'Completeness score: {comp_score:.3f}')
    return model


def main():
    #print("in main")
    parser = argparse.ArgumentParser(description='Cluster 20 newsgroup dataset.')
    parser.add_argument('--ncluster', nargs='+', type=int, default=[20], help='Number of clusters.')
    parser.add_argument('--kmeans', action='store_true', help='Use KMeans clustering.')
    parser.add_argument('--whc', action='store_true', help='Use Ward Hierarchical Clustering.')
    parser.add_argument('--ac', action='store_true', help='Use agglomerative clustering.')
    parser.add_argument('--dbscan', action='store_true', help='Use DBSCAN clustering.')

    #print(parser.parse_args())
    args = parser.parse_args()
    # Load data
    data, labels = load_data()
    #print("labels in main ",labels)
    # Cluster documents
    if args.kmeans:
        cluster_documents(data, labels, args.ncluster, 'kmeans')
    elif args.whc:
        cluster_documents(data, labels, args.ncluster, 'whc')
    elif args.ac:
        cluster_documents(data, labels, args.ncluster, 'ac')
    elif args.dbscan:
        cluster_documents(data, labels, args.ncluster, 'dbscan')
   
if __name__ == '__main__':
    main()