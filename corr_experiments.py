import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import numpy as np
import seaborn as sns
PATH_DEEZER = "./datasets/deezer_clean_data/HR_edges.csv"
PATH_TWITCH = "./datasets/twitch/DE/musae_DE_edges.csv"
PATH_CACONDMAT = "./datasets/CA-CondMat.txt"
SAMPLE_SIZE = 20
ITERATIONS = 20

def import_graph(file_path, file_format="edgelist"):
    """
    Import a graph dataset from a file.

    Parameters:
        file_path (str): Path to the file containing the graph dataset.
        file_format (str): Format of the graph file. Options are "edgelist", "adjlist", "gml", "graphml", "pajek".

    Returns:
        networkx.Graph: The imported graph.
    """
    if file_format == "edgelist":
        graph = nx.read_edgelist(file_path, delimiter = ",")
    elif file_format == "adjlist":
        graph = nx.read_adjlist(file_path)
    elif file_format == "gml":
        graph = nx.read_gml(file_path)
    elif file_format == "graphml":
        graph = nx.read_graphml(file_path)
    elif file_format == "pajek":
        graph = nx.read_pajek(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return graph

def spearman_rank_correlation(rank_1, rank_2):
    """
    Calculate Spearman's rank correlation coefficient for two vectors of (node, score).

    Parameters:
        rank_1 (list of tuples): List of (node, score) tuples, sorted by score.
        rank_2 (list of tuples): List of (node, score) tuples, sorted by score.

    Returns:
        float: Spearman's rank correlation coefficient.
    """
    # Extract node orderings from rank_1 and rank_2
    nodes_1 = [node for node, _ in rank_1]
    nodes_2 = [node for node, _ in rank_2]

    # Check that the node sets are identical
    if set(nodes_1) != set(nodes_2):
        raise ValueError("The node sets in rank_1 and rank_2 must be identical.")

    # Create node-to-rank mappings for both rank_1 and rank_2
    rank_map_1 = {node: rank for rank, node in enumerate(nodes_1, start=1)}
    rank_map_2 = {node: rank for rank, node in enumerate(nodes_2, start=1)}

    # Extract ranks for each node in the same order
    ranks_1 = [rank_map_1[node] for node in nodes_1]
    ranks_2 = [rank_map_2[node] for node in nodes_1]

    # Calculate Spearman's rank correlation coefficient
    n = len(ranks_1)
    if n == 0:
        raise ValueError("The input vectors must not be empty.")

    # Compute the sum of squared rank differences
    d_squared_sum = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks_1, ranks_2))

    # Apply the Spearman rank correlation formula
    if n > 1:
        spearman_coefficient = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    else:
        spearman_coefficient = 1

    return spearman_coefficient

def kendalls_tau(rank_1, rank_2):
    """
    Calculate Kendall's Tau correlation coefficient for two vectors of (node, score).

    Parameters:
        rank_1 (list of tuples): List of (node, score) tuples, sorted by score.
        rank_2 (list of tuples): List of (node, score) tuples, sorted by score.

    Returns:
        float: Kendall's Tau correlation coefficient.
    """
    # Extract node orderings from rank_1 and rank_2
    nodes_1 = [node for node, _ in rank_1]
    nodes_2 = [node for node, _ in rank_2]

    # Create node-to-rank mappings for both rank_1 and rank_2
    rank_map_1 = {node: rank for rank, node in enumerate(nodes_1, start=1)}
    rank_map_2 = {node: rank for rank, node in enumerate(nodes_2, start=1)}

    # Extract ranks for each node in the same order
    ranks_1 = [rank_map_1[node] for node in nodes_1]
    ranks_2 = [rank_map_2[node] for node in nodes_1]

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    n = len(ranks_1)

    if n < 2:
        return 1

    for i in range(n):
        for j in range(i + 1, n):
            rank_diff_1 = ranks_1[i] - ranks_1[j]
            rank_diff_2 = ranks_2[i] - ranks_2[j]

            if rank_diff_1 * rank_diff_2 > 0:
                concordant += 1
            elif rank_diff_1 * rank_diff_2 < 0:
                discordant += 1

    # Compute Kendall's Tau
    tau = (concordant - discordant) / (0.5 * n * (n - 1))

    return tau

def plot_rankings(rank_1, rank_2):
    """
    Plot rankings with nodes from rank_1 on the x-axis and score on the y-axis for both rankings.

    Parameters:
        rank_1 (list of tuples): List of (node, score) tuples, sorted by score.
        rank_2 (list of tuples): List of (node, score) tuples, sorted by score.
    """
    nodes = [node for node, _ in rank_1]
    n = len(nodes)
    scores_1 = [score for _, score in rank_1]
    scores_2_dict = dict(rank_2)
    scores_2 = [scores_2_dict[node] for node in nodes]

    plt.figure(figsize=(8, 6))
    plt.plot(range(n), scores_1, marker='o', label='Complete ranking')
    plt.plot(range(n), scores_2, marker='s', label='Truncated ranking')

    plt.xlabel('Nodes')
    plt.ylabel('Score')
    plt.title('Node vs Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

def graph_features(graph):
    """
    Compute an overview of a graph's features, including edge density.

    Parameters:
        graph (networkx.Graph): The input graph.

    Returns:
        dict: A dictionary with graph features.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edge_density = nx.density(graph)

    features = {
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Edge Density": edge_density
    }

    return features

def truncated_harmonic_centrality(graph, nbunch, radius):
    centralities = list()
    for u in nbunch:
        centralities.append(nx.harmonic_centrality(nx.ego_graph(graph, u, radius)).get(u,0))
    return centralities
    
def compute_subgraph_centralities(graph, center, sample_size):
    """
    Compute the harmonic centrality of a subgraph with n nodes.

    Parameters:
        graph (networkx.Graph): The original graph.
        n (int): Number of nodes in the subgraph.

    Returns:
        dict: A dictionary with harmonic centralities and truncated centralities.
    """
    # Select a subgraph with n random nodes
    sampled_nodes = bfs_first_h_nodes(graph, center, sample_size)
    #print(f"Sampled nodes = {sampled_nodes}")
    
    # Compute harmonic centrality for nodes in the original graph
    scores = nx.harmonic_centrality(graph, nbunch=sampled_nodes)
    t_harmonic_centrality = truncated_harmonic_centrality(graph, nbunch=sampled_nodes, radius=2)

    t_scores = dict(zip(sampled_nodes,t_harmonic_centrality))

    return scores, t_scores

def bfs_first_h_nodes(graph, start_node, h):
    """
    Perform a breadth-first search (BFS) on a graph and return the first h nodes explored.

    Parameters:
        graph (networkx.Graph): The input graph.
        start_node (node): The starting node for the BFS.
        h (int): The number of nodes to explore.

    Returns:
        list: A list of the first h nodes explored.
    """
    visited = []
    queue = [start_node]

    while queue and len(visited) < h:
        current_node = queue.pop(0)
        if current_node not in visited:
            visited.append(current_node)
            neighbors = list(graph.neighbors(current_node))
            queue.extend(neighbors)

    return visited[:h]

def generate_and_save_heatmap(data: np.ndarray, xticklabels: np.ndarray, yticklabels: np.ndarray, filename: str):
    """
    Generates a heatmap from a 2D numpy array and saves it as an image file with the given filename.
    
    Parameters:
    - data (np.ndarray): 2D NumPy array representing the data.
    - filename (str): The name of the file where the heatmap image will be saved.
    """
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=False, xticklabels=xticklabels, yticklabels=yticklabels, cmap='viridis', cbar=True)

    # Save the heatmap as an image file with the provided filename
    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.close()

def experiments_gnp(n,p,d):
    """
    Parameters:
    - n: list() of int
    - p: list() of floats (probabilities)
    - d: int, is the number of tries for every couple in (n,p)
    """
    r = len(p)
    c = len(n)
    corr_matrix = np.zeros((r,c,d))
    kendall_matrix = np.zeros((r,c,d))
    for n_i, n_val in enumerate(n):
        for p_i, p_val in enumerate(p):
            for i in range(d):
                print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{p} | Iteration: {i+1}/{d}")
                graph = nx.gnp_random_graph(n_val, p_val)
                node = sample(list(graph.nodes()), 1)[0]
                centralities, t_centralities = compute_subgraph_centralities(graph, node, sample_size = SAMPLE_SIZE)
                rank = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
                t_rank = sorted(t_centralities.items(), key=lambda x: x[1], reverse=True)
                spearman_rank_corr_val = spearman_rank_correlation(rank, t_rank)
                kendalls_tau_val = kendalls_tau(rank, t_rank)
                corr_matrix[p_i, n_i, i] = spearman_rank_corr_val
                kendall_matrix[p_i, n_i, i] = kendalls_tau_val
    np.save("output_corr", corr_matrix)
    np.save("output_kendall", kendall_matrix)
    avg_corr_matrix = np.mean(corr_matrix, axis=2)
    std_corr_matrix = np.std(corr_matrix, axis=2)
    avg_kendall_matrix = np.mean(kendall_matrix, axis=2)
    std_kendall_matrix = np.std(kendall_matrix, axis=2)
    generate_and_save_heatmap(avg_corr_matrix, xticklabels=n, yticklabels=p, filename="avg_corr_matrix")
    generate_and_save_heatmap(std_corr_matrix, xticklabels=n, yticklabels=p, filename="std_corr_matrix")
    generate_and_save_heatmap(avg_kendall_matrix, xticklabels=n, yticklabels=p, filename="avg_kendall_matrix")
    generate_and_save_heatmap(std_kendall_matrix, xticklabels=n, yticklabels=p, filename="std_kendall_matrix")

def experiments_watts_strogatz(k,n,p,d):
    """
    Parameters:
    - n: list() of int
    - p: list() of floats (probabilities)
    - d: int, is the number of tries for every couple in (n,p)
    """
    r = len(p)
    c = len(n)
    for k_val in k:
        print(f"Value of k = {k_val} ")
        corr_matrix = np.zeros((r,c,d))
        kendall_matrix = np.zeros((r,c,d))
        for n_i, n_val in enumerate(n):
            for p_i, p_val in enumerate(p):
                for i in range(d):
                    print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{p} | Iteration: {i+1}/{d}")
                    graph = nx.gnp_random_graph(n_val, p_val)
                    node = sample(list(graph.nodes()), 1)[0]
                    centralities, t_centralities = compute_subgraph_centralities(graph, node, sample_size = SAMPLE_SIZE)
                    rank = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
                    t_rank = sorted(t_centralities.items(), key=lambda x: x[1], reverse=True)
                    spearman_rank_corr_val = spearman_rank_correlation(rank, t_rank)
                    kendalls_tau_val = kendalls_tau(rank, t_rank)
                    corr_matrix[p_i, n_i, i] = spearman_rank_corr_val
                    kendall_matrix[p_i, n_i, i] = kendalls_tau_val
        corr_name = "output_corr_k=" + str(k_val) 
        kendall_name = "output_kendall_k" + str(k_val)
        filename_1 = "avg_corr_matrix_k" + str(k_val) 
        filename_2 = "std_corr_matrix" + str(k_val)
        filename_3 = "avg_kendall_matrix" + str(k_val)
        filename_4 = "std_kendall_matrix" + str(k_val)
        np.save(corr_name, corr_matrix)
        np.save(kendall_name, kendall_matrix)
        avg_corr_matrix = np.mean(corr_matrix, axis=2)
        std_corr_matrix = np.std(corr_matrix, axis=2)
        avg_kendall_matrix = np.mean(kendall_matrix, axis=2)
        std_kendall_matrix = np.std(kendall_matrix, axis=2)
        generate_and_save_heatmap(avg_corr_matrix, xticklabels=n, yticklabels=p, filename=filename_1)
        generate_and_save_heatmap(std_corr_matrix, xticklabels=n, yticklabels=p, filename=filename_2)
        generate_and_save_heatmap(avg_kendall_matrix, xticklabels=n, yticklabels=p, filename=filename_3)
        generate_and_save_heatmap(std_kendall_matrix, xticklabels=n, yticklabels=p, filename=filename_4)

def experiments_regular_graphs(n,degree,d):
    """
    Parameters:
    - n: list() of int
    - p: list() of floats (probabilities)
    - d: int, is the number of tries for every couple in (n,p)
    """
    r = len(degree)
    c = len(n)
    
    corr_matrix = np.zeros((r,c,d))
    kendall_matrix = np.zeros((r,c,d))
    for n_i, n_val in enumerate(n):
        for p_i, p_val in enumerate(degree):
            if p_val == "$":
                p_val = 1
            elif p_val == 1/100 and n_val == 100:
                print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{degree} | Iteration: instantly completed")
                corr_matrix[p_i, n_i,:] = corr_matrix[p_i-1, n_i,:]
                kendall_matrix[p_i, n_i,:] = kendall_matrix[p_i-1, n_i,:]
                continue
            else:
                p_val = int(p_val * n_val)
            for i in range(d):
                print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{degree} | Iteration: {i+1}/{d}")
                graph = nx.random_regular_graph(p_val, n_val)
                node = sample(list(graph.nodes()), 1)[0]
                centralities, t_centralities = compute_subgraph_centralities(graph, node, sample_size = SAMPLE_SIZE)
                rank = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
                t_rank = sorted(t_centralities.items(), key=lambda x: x[1], reverse=True)
                spearman_rank_corr_val = spearman_rank_correlation(rank, t_rank)
                kendalls_tau_val = kendalls_tau(rank, t_rank)
                corr_matrix[p_i, n_i, i] = spearman_rank_corr_val
                kendall_matrix[p_i, n_i, i] = kendalls_tau_val
    corr_name = "output_corr_regular" 
    kendall_name = "output_kendall_regular"
    filename_1 = "avg_corr_matrix_regular" 
    filename_2 = "std_corr_regular" 
    filename_3 = "avg_kendall_regular"
    filename_4 = "std_kendall_regular"
    np.save(corr_name, corr_matrix)
    np.save(kendall_name, kendall_matrix)
    avg_corr_matrix = np.mean(corr_matrix, axis=2)
    std_corr_matrix = np.std(corr_matrix, axis=2)
    avg_kendall_matrix = np.mean(kendall_matrix, axis=2)
    std_kendall_matrix = np.std(kendall_matrix, axis=2)
    generate_and_save_heatmap(avg_corr_matrix, xticklabels=n, yticklabels=degree, filename=filename_1)
    generate_and_save_heatmap(std_corr_matrix, xticklabels=n, yticklabels=degree, filename=filename_2)
    generate_and_save_heatmap(avg_kendall_matrix, xticklabels=n, yticklabels=degree, filename=filename_3)
    generate_and_save_heatmap(std_kendall_matrix, xticklabels=n, yticklabels=degree, filename=filename_4)

def experiments_barabasi_albert(n,m,d):
    """
    Parameters:
    - n: list() of int
    - p: list() of floats (probabilities)
    - d: int, is the number of tries for every couple in (n,p)
    """
    r = len(m)
    c = len(n)
    
    corr_matrix = np.zeros((r,c,d))
    kendall_matrix = np.zeros((r,c,d))
    for n_i, n_val in enumerate(n):
        for p_i, p_val in enumerate(m):
            if p_val == "$":
                p_val = 1
            elif p_val == 1/100 and n_val == 100:
                print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{m} | Iteration: instantly completed")
                corr_matrix[p_i, n_i,:] = corr_matrix[p_i-1, n_i,:]
                kendall_matrix[p_i, n_i,:] = kendall_matrix[p_i-1, n_i,:]
                continue
            else:
                p_val = int(p_val * n_val)
            for i in range(d):
                print(f"Number of nodes: {n_val}/{n} | Probabilities: {p_val}/{m} | Iteration: {i+1}/{d}")
                graph = nx.barabasi_albert_graph(n_val, p_val)
                node = sample(list(graph.nodes()), 1)[0]
                centralities, t_centralities = compute_subgraph_centralities(graph, node, sample_size = SAMPLE_SIZE)
                rank = sorted(centralities.items(), key=lambda x: x[1], reverse=True)
                t_rank = sorted(t_centralities.items(), key=lambda x: x[1], reverse=True)
                spearman_rank_corr_val = spearman_rank_correlation(rank, t_rank)
                kendalls_tau_val = kendalls_tau(rank, t_rank)
                corr_matrix[p_i, n_i, i] = spearman_rank_corr_val
                kendall_matrix[p_i, n_i, i] = kendalls_tau_val
    corr_name = "output_corr_barabasi_albert" 
    kendall_name = "output_kendall_barabasi_albert"
    filename_1 = "avg_corr_matrix_barabasi_albert" 
    filename_2 = "std_corr_barabasi_albert" 
    filename_3 = "avg_kendall_barabasi_albert"
    filename_4 = "std_kendall_barabasi_albert"
    np.save(corr_name, corr_matrix)
    np.save(kendall_name, kendall_matrix)
    avg_corr_matrix = np.mean(corr_matrix, axis=2)
    std_corr_matrix = np.std(corr_matrix, axis=2)
    avg_kendall_matrix = np.mean(kendall_matrix, axis=2)
    std_kendall_matrix = np.std(kendall_matrix, axis=2)
    generate_and_save_heatmap(avg_corr_matrix, xticklabels=n, yticklabels=m, filename=filename_1)
    generate_and_save_heatmap(std_corr_matrix, xticklabels=n, yticklabels=m, filename=filename_2)
    generate_and_save_heatmap(avg_kendall_matrix, xticklabels=n, yticklabels=m, filename=filename_3)
    generate_and_save_heatmap(std_kendall_matrix, xticklabels=n, yticklabels=m, filename=filename_4)


if __name__ == "__main__":
    #corr_matrix, kendall_matrix = experiments_gnp(n = [100, 1000], p = [0.01, 0.1], d = 10)
    
    experiments_gnp(n = [100,500,1000,2500], p = [0.001, 0.01, 0.1], d = ITERATIONS)
    experiments_watts_strogatz(k = [1,2,3,5,10], n = [100,500,1000,2500], p = [0.001, 0.01, 0.1], d = ITERATIONS)
    experiments_regular_graphs(n = [100,500,1000,2500], degree = ["$",1/100,1/50,1/25,2/25,4/25,8/25], d = ITERATIONS)
    experiments_barabasi_albert(n = [100,500,1000,2500], m = ["$",1/100,1/50,1/25,2/25,4/25,8/25], d = ITERATIONS)
    
    """
    corr_matrix = np.load("output_corr.npy")
    kendall_matrix = np.load("output_kendall.npy")
    avg_corr_matrix = np.mean(corr_matrix, axis=2)
    std_corr_matrix = np.std(corr_matrix, axis=2)
    avg_kendall_matrix = np.mean(kendall_matrix, axis=2)
    std_kendall_matrix = np.std(kendall_matrix, axis=2)
    generate_and_save_heatmap(avg_corr_matrix, xticklabels=[100,500,1000,2500], yticklabels=[0.001, 0.01, 0.1], filename="avg_corr_matrix")
    generate_and_save_heatmap(std_corr_matrix, xticklabels=[100,500,1000,2500], yticklabels=[0.001, 0.01, 0.1], filename="std_corr_matrix")
    generate_and_save_heatmap(avg_kendall_matrix, xticklabels=[100,500,1000,2500], yticklabels=[0.001, 0.01, 0.1], filename="avg_kendall_matrix")
    generate_and_save_heatmap(std_kendall_matrix, xticklabels=[100,500,1000,2500], yticklabels=[0.001, 0.01, 0.1], filename="std_kendall_matrix")
    """


