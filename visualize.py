from graphviz import Digraph
from my_ml_lib.nn import get_all_nodes_and_edges



def draw_dot(root, format='png', filename='graph', directory='.', rankdir='LR'):
    """
    Visualize the computation graph starting from `root` (Value object).

    Parameters
    ----------
    root : Value
        Root node of computation graph
    format : str
        Output format (png, svg, etc.)
    filename : str
        Output file name (without extension)
    directory : str
        Directory where the image will be saved
    rankdir : str
        Graph orientation ('LR' = left-right, 'TB' = top-bottom)
    """

    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    nodes, edges = get_all_nodes_and_edges(root)

    for n in nodes:
        uid = str(id(n))
        label = f"{n.label or ''} | data={n.data:.4f} | grad={n.grad:.4f}"
        dot.node(uid, label, shape='record')

        if n.op:
            op_id = uid + n.op
            dot.node(op_id, n.op)
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    out_path = dot.render(filename=filename, directory=directory, cleanup=True)
    print(f"Graph saved at: {out_path}")
    return dot

