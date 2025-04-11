# PrivMedRAG

Authors: Haoli Yin and Tuktu Doga Nazli

PrivMedRAG is a retrieval-augmented generation (RAG) system for privacy-preserving medical question generation and evaluation.

Note that currently PrivMedRAG only supports ubuntu.

## Setup

We manage dependencies using `uv` so please install it first then run the following command to install all dependencies.

```bash
uv sync
```

### FAQ
- You might run into `dgl` dependency install issues and will have to install from source. Note that installation currently only supported on ubuntu, not on Apple Silicon nor Windows.
```bash 
git clone --recursive https://github.com/dmlc/dgl.git
cd dgl
mkdir build
cd build
cmake -DUSE_CUDA=OFF \
  -DUSE_OPENMP=ON \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  ..
make -j4
cd ../python
pip install -e .
```


## Data

### PrimeKG download

```bash
mkdir -p dataset/primekg
wget -O dataset/primekg/raw/edges.csv https://dataverse.harvard.edu/api/access/datafile/6180616
wget -O dataset/primekg/raw/nodes.tsv https://dataverse.harvard.edu/api/access/datafile/6180617
```

### Synthetic MCQ Generation

To prepare to the data for synthetic MCQ generation:

```bash
wget -O dataset/synthea-dataset-100.zip https://github.com/lhs-open/synthetic-data/raw/main/record/synthea-dataset-100.zip
unzip dataset/synthea-dataset-100.zip -d dataset/synthea-dataset-100
```

To generate the synthetic data and run evaluations, please see: [./evaluate/README.md](./evaluate/README.md).

## Acknowledgements

We build off the work of RGL as seen in [README_RGL.md](README_RGL.md).