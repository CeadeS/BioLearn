# Synaptic Diversity: Bridging Biological and Artificial Neural Networks

## Introduction

Welcome to the reproduction repository for "Synaptic Diversity: Concept Transfer from Biological to Artificial Neural Networks" by Martin Hofmann, Moritz Franz Peter Becker, Christian Tetzlaff, and Patrick Mäder. This groundbreaking research explores how principles from biological neural networks can enhance artificial neural networks.

## Project Overview

In biological brains, synapses - the connections between neurons - exhibit remarkable diversity and plasticity. Our project aims to transfer these biological concepts to artificial neural networks (ANNs) to improve their learning capabilities and robustness. We focus on three key mechanisms:

1. **Diversity in Synaptic Plasticity**: Not all synapses learn at the same rate in biological systems.
2. **Spontaneous Spine Remodeling**: Biological synapses can form and disappear dynamically.
3. **Multi-Synaptic Connectivity**: Multiple connections can exist between neuron pairs in biological networks.

## Key Methods

Our implementation introduces several biologically-inspired methods to standard ANNs:

1. `fuzzy_learning_rates`: 
   - Mimics diverse synaptic plasticity by applying random, constant factors to weight gradients.
   - Each synapse (weight) has its own learning rate modifier, allowing for heterogeneous learning across the network.

2. `weight_rejuvenation`: 
   - Simulates spontaneous spine remodeling by randomly reinitializing certain weights.
   - Weights are selected for rejuvenation based on their current magnitude, with smaller weights more likely to be reset.

3. `weight_splitting`: 
   - Implements multi-synaptic connectivity by allowing multiple weights between neuron pairs.
   - This is achieved by duplicating and summing weights, increasing the network's expressive power.

These methods work together to create a more biologically plausible learning process, potentially leading to more robust and efficient ANNs.

## Installation

It is usually reasonable but not mendatory, in most cases, to create a virtual environment for example using conda.

```bash
conda create -n biolearn python=3.8
conda activate biolearn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

You can easily install our `pip install pytorch_bio_transformations` package via PyPI:

```bash
pip install pytorch_bio_transformations
```

or download it from [GitHub](https://github.com/CeadeS/pytorch_bio_transformations.git).
Our [Documentation](https://ceades.github.io/pytorch_bio_transformations/index.html) provides additional information.

## Usage

Here's a quick example of how to use our bio-inspired transformations:

```python
from bio_transformations import BioConverter
from bio_transformations.bio_config import BioConfig
from torch.nn import functional as F
import torch.nn as nn

# Define your neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Create a BioConverter with custom settings
config = BioConfig(
    fuzzy_learning_rate_factor_nu=0.2,
    rejuvenation_parameter_dre=10.0,
    weight_splitting_Gamma=2
)
converter = BioConverter(config=config)

# Convert your network
bio_network = converter(MyNetwork)()

# Use bio_network as you would a regular PyTorch model
```

## Reproducing Our Experiments

```bash
pip install pytorch_bio_transformations
pip install -r eval/requirements.txt
```

### Evaluating Accuracy and Learning Progress

To reproduce our accuracy and learning progress experiments:
1. Go to the eval directory:
   ```bash
   cd eval/
   ```
1. Run the experiment script:
   ```bash
   python run_experiment.py
   ```

### Gradient Reconstruction Experiments

To reproduce our gradient reconstruction experiments:

1. Train a model:
   - Open and run the notebook `eval/train_models_for_gradinversion.ipynb`
   - The trained model will be saved in `eval/models`

2. Perform gradient inversion:
   - Open and run the notebook `grad_rec/SimpleReconstruction.ipynb`

## Results

Our experiments demonstrate that incorporating these biologically-inspired modifications leads to:

- Faster learning rates in various tasks
- Improved prediction accuracy, especially in complex datasets
- Enhanced resilience against gradient inversion attacks, improving privacy

For a detailed analysis of our observations, please refer to our paper.

## Citation

If you use this code or our methods in your research, please cite our paper:

```bibtex
@article{hofmann2023synaptic,
  title={Synaptic Diversity: Concept Transfer from Biological to Artificial Neural Networks},
  author={Hofmann, Martin and Becker, Moritz Franz Peter and Tetzlaff, Christian and Mäder, Patrick},
  journal={Journal Name},
  year={2023},
  publisher={Publisher}
}
```

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/CeadeS/pytorch_bio_transformations) file for guidelines on how to contribute to the pytorch_bio_transformations project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please open an issue in this repository or contact the authors directly at Martin.Hofmann@tu-ilmenau.de.

We're excited to see how the community uses and builds upon these bio-inspired methods to advance the field of artificial neural networks!