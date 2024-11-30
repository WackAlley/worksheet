import graphviz
from forward_backward_functions_and_nodes import Expr_end_node
import traceback

def print_graph(node, graph, parent_id=""):
    
    node_id = str(id(node))
    if isinstance(node, Expr_end_node):
        node_label = str(node.instance)
    else:
        if node.func_name == 'Multiply_scalar':
            node_label = node.func_name # scalar value would be nice
        else:
            node_label = node.func_name
    graph.node(node_id, node_label)  
    
    if parent_id:
        graph.edge(parent_id, node_id)
        
    if hasattr(node, 'childs') and node.childs:
            for child in node.childs:
                print_graph(child, graph, node_id)#, constraint='false')