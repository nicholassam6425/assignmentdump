from copy import deepcopy
import argparse

class Node:
    pr: float
    num_links_to: int
    linked_by: None

node_list = {}
parser = argparse.ArgumentParser(
    prog='PageRank',
    description='Calculates PageRank for the 2002 Stanford web graph'
)
parser.add_argument('--maxiteration', type=int, help="The maximum number of iterations to stop if algorithm has not converged", default=20)
parser.add_argument('--lambda', dest='lmbda', type=float, help="The lambda parameter value", default=0.25)
parser.add_argument('--thr', type=float, help="The threshold value", default=0)
parser.add_argument('--nodes', type=str, help="The NodeIDs that we want to get their PageRank values at the end of iterations", default='1')
args = parser.parse_args()
with open("web-Stanford.txt", "r") as file:
    lines = file.readlines()
    doc_info = lines[2].split()
    node = Node()
    node.pr = 1/int(doc_info[2])
    node.num_links_to = 0
    node.linked_by = []
    num_nodes = int(doc_info[2])
    for i in range(1, int(doc_info[2])+1):
        node_list[i] = deepcopy(node)

    for line in lines:
        info = line.split()
        if info[0] != "#":
            if int(info[0]) not in node_list[int(info[1])].linked_by:
                node_list[int(info[1])].linked_by.append(int(info[0]))
            node_list[int(info[0])].num_links_to += 1
    
iterations = 0
next_node_list = {}
threshold = False
while (iterations < args.maxiteration):
    for i in node_list:
        node = deepcopy(node_list[i])
        sigma = 0
        for k in node.linked_by:
            sigma += node_list[k].pr/node_list[k].num_links_to
        node.pr = (args.lmbda/num_nodes) + (1-args.lmbda) * sigma
        #if threshold is met, continue calculating the rest of this iteration
        if(abs(node.pr - node_list[i].pr) < args.thr):
            if not threshold:
                print(f'Threshold reached at node {i} at iteration {iterations}')
                threshold = True
            iterations = args.maxiteration-1
        next_node_list[i] = node
    node_list = deepcopy(next_node_list)
    iterations += 1
if not threshold:
    print("Reached max iterations")
printnodes = args.nodes.strip('][').split(',')
for i in printnodes:
    print(f'Node {i} PageRank: {node_list[int(i)].pr}')