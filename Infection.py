"""
This code describes the explanation for GCN model using GNNExplainer algorithm over InfectionDataset dataset.

We first train the simple GCN model. And record the results.

After that, we explain this model, and use two metrics to evaluate the explanation results. We also record the results.
"""


import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch_geometric.datasets import InfectionDataset
from torch_geometric.datasets.graph_generator import ERGraph


import torch_geometric.transforms as T
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain import groundtruth_metrics
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph

dataset = InfectionDataset(
    graph_generator=ERGraph(num_nodes=500, edge_prob=0.004),
    num_infected_nodes=50,
    max_path_length=3,
)

data = dataset[0]
print(data.num_node_features)
print(dataset.num_classes)

idx = torch.arange(data.num_nodes)

train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=data.y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(data.num_node_features, hidden_channels=35, num_layers=3,
            out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)



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
results for node classification:
Loss: 0.4698, Train: 0.8175, Test: 0.8700
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

    # Explanation acc and auroc over all nodes:
    targets, preds = [], []

    node_indices = range(0, data.num_nodes)
    for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
        target = data.y if explanation_type == 'phenomenon' else None
        explanation = explainer(data.x, data.edge_index, index=node_index,
                                target=target)
        
        
        
        '''
        Find the k-hop-subgraph of given node, defined as v.  hard_edge_mask indicates which edges were preserved. 
        Details could been seen 
        https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.k_hop_subgraph

        Q: Why the num_hops is selected to 3?
        A: In the dataset part, max_path_length: The maximum shortest path length to determine whether a node will be infected. 
        Cause the max_path_length is set to 3, so in the explanation part, we set the num_hops to 3, consistent to the dataset setting.
        '''
        _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops=3,
                                                 edge_index=data.edge_index)
        
        targets.append(data.edge_mask[hard_edge_mask].cpu())
        preds.append(explanation.edge_mask[hard_edge_mask].cpu())

        """
        In the PyG documentation, the description for the dataset is as follows:

        The dataset describes a node classification task of predicting the length of the shortest path to infected nodes, 
        with corresponding ground-truth edge-level masks.

        We decide to compare the prediction edge mask with the ground-truth edge mask. Here, we select two metrics, accuracy and auroc.
        And the results are shown behind.
        """

    """
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/explain/metric/basic.html#groundtruth_metrics
    """
    # acc = groundtruth_metrics(torch.cat(preds), torch.cat(targets), 'accuracy')
    # print(f'acc (explanation type {explanation_type:10}): {acc:.4f}')
    """
    results for acc:

    phenomenon: 0.4114
    model: 0.4046
    """


    auroc = groundtruth_metrics(torch.cat(preds), torch.cat(targets), 'auroc')
    print(f'auroc (explanation type {explanation_type:10}): {auroc:.4f}')
    '''
    results for auroc
    phenomenon: 0.4496
    model: 0.4421
    '''
    
