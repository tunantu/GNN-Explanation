"""
This code describes the explanation for a simple GCN model using GNNExplainer algorithm over BAShapes dataset.

We first train the simple GCN model and recode its results.

After that, we explain this model and evaluate it using two metrics. In the description about BAShape dataset:

Ground-truth node-level and edge-level explainabilty masks are given based on whether nodes and edges are part of a certain motif or not.

But so far I just figured out the comparison between ground-truth edge mask and prediction edge mask. I will try to finish another one.

And We record  the results behind.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain import groundtruth_metrics
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph

dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=5),
    motif_generator='house',
    num_motifs=80,
    transform=T.Constant(),
)
data = dataset[0]

idx = torch.arange(data.num_nodes)
train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(data.num_node_features, hidden_channels=20, num_layers=3,
            out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)


"""
In the BAShape dataset, nodes are of type 0 on the base graph, 
while the top, middle, and bottom parts of the "house" are categorized as types 1-3 respectively.
"""

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


pbar = tqdm(range(1, 2001))
for epoch in pbar:
    loss = train()
    if epoch == 1 or epoch % 200 == 0:
        train_acc, test_acc = test()
        pbar.set_description(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                             f'Test: {test_acc:.4f}')
pbar.close()
model.eval()

'''
The results for this simple GCN model:
Loss: 0.6302, Train: 0.8571, Test: 0.8500
'''

for explanation_type in ['phenomenon', 'model']:
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=300),
        explanation_type=explanation_type,
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    # Explanation ACC and roauc over all test nodes:
    node_targets, node_preds = [], [] # haven't been finished.
    edge_targets, edge_preds = [], []


    '''
    In the BAShape dataset, nodes are of type 0 on the base graph, and there are 300 nodes in the basic graph. 
    We are focusing on explaining the "house", so the starting node is chosen as 300. 
    Since each house has 5 nodes, the interval is 5. 
    By explaining the first node of each house, we can explain all 80 houses.
    '''
    node_indices = range(300, data.num_nodes, 5)
    for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
        target = data.y if explanation_type == 'phenomenon' else None
        explanation = explainer(data.x, data.edge_index, index=node_index,
                                target=target)
        
        
        
        '''
        Find the k-hop-subgraph of given node, defined as v.  hard_edge_mask indicates 
        which edges were preserved. 
        Details could been seen 
        https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.k_hop_subgraph
        '''
        _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=3,
                                                 edge_index=data.edge_index)
        
        edge_targets.append(data.edge_mask[hard_edge_mask].cpu())
        edge_preds.append(explanation.edge_mask[hard_edge_mask].cpu())





    """
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/explain/metric/basic.html#groundtruth_metrics
    """
    acc = groundtruth_metrics(torch.cat(edge_preds), torch.cat(edge_targets), 'accuracy')
    print(f'acc (explanation type {explanation_type:10}): {acc:.4f}')

    auroc = groundtruth_metrics(torch.cat(edge_preds), torch.cat(edge_targets), 'auroc')
    print(f'auroc (explanation type {explanation_type:10}): {auroc:.4f}')

'''
results for acc:
phenomenon: 0.9814
model: 0.9814

results for auroc:
phenomenon: 0.9834
model: 0.9831

'''