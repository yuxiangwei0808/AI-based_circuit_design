import networkx as nx
from itertools import product


def _expand(G, explored_nodes, explored_edges):
    """
    Expand existing solution by a process akin to BFS.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    explored_nodes: set of ints
        nodes visited

    explored_edges: set of 2-tuples
        edges visited

    Returns:
    --------
    solutions: list, where each entry in turns contains two sets corresponding to explored_nodes and explored_edges
        all possible expansions of explored_nodes and explored_edges

    """
    frontier_nodes = list()
    frontier_edges = list()
    for v in explored_nodes:
        for u in nx.neighbors(G,v):
            if not (u in explored_nodes):
                frontier_nodes.append(u)
                frontier_edges.append([(u, v), (v, u)])

    return zip([explored_nodes | frozenset([v]) for v in frontier_nodes], [explored_edges | frozenset(e) for e in frontier_edges])


def find_all_spanning_trees(G):
    """
    Find all spanning trees of a Graph.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph

    Returns:
    ST: list of networkx.Graph() instances
        list of all spanning trees

    """
    root = list(G.nodes())[0]
    # initialise solution
    explored_nodes = frozenset([root])
    explored_edges = frozenset([])
    solutions = [(explored_nodes, explored_edges)]
    # we need to expand solutions number_of_nodes-1 times
    for ii in range(G.number_of_nodes()-1):
        # get all new solutions
        solutions = [_expand(G, nodes, edges) for (nodes, edges) in solutions]
        # flatten nested structure and get unique expansions
        solutions = set([item for sublist in solutions for item in sublist])

    return [nx.from_edgelist(edges) for (nodes, edges) in solutions]


def validate_two_tree(g, excluded_pairs):
    for p in excluded_pairs:
        if nx.has_path(g, p[0], p[1]):
            return 0
    return 1


def find_all_two_trees(G, trees: list, node_pair: list):
    """
    Find all two trees of a Graph.

    Arguments:
    ----------
    G: networkx.Graph() instance
        full graph
    tree: list object
        contains found trees
    node_pair: list
        contains node pairs that needs to be separated

    Returns:
    ST: list of networkx.Graph() instances
        list of all spanning trees

    """
    excluded_pair = list(product(node_pair[0], node_pair[1]))
    excluded_pair = [sorted(x) for x in excluded_pair]
    valid_trees = []
    for tree in trees:
        edges = list(tree.edges())
        flag = [True for e in edges if sorted(e) in excluded_pair]
        if not len(flag):
            for edge in edges:
                tmp = tree.copy()
                tmp.remove_edge(edge[0], edge[1])
                fg = validate_two_tree(tmp, excluded_pair)
                if fg:
                    valid_trees.append(tmp)
    return valid_trees


if __name__ == '__main__':
    G = nx.petersen_graph()
    tree = find_all_spanning_trees(G)
    print(1)