# PrivMedRAG

Authors: Haoli Yin and Tuktu Doga Nazli

PrivMedRAG is a retrieval-augmented generation (RAG) system for privacy-preserving medical question generation and evaluation.

**NOTE**: PrivMedRAG currently only supports Ubuntu.

## Setup

### Prerequisites

1. Clone the repository
```bash
git clone --recursive https://github.com/Nano1337/privmedrag.git
cd privmedrag
```

2. We manage dependencies using `uv` so please install it first using the docs [here](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it.

3. Then run the following command to install base dependencies.

```bash
uv sync
source .venv/bin/activate
```


4. You will have to install `dgl` from source.Note that installation currently only supported on ubuntu, not on Apple Silicon nor Windows.

Please be patient as this build may take up to 10 minutes.

```bash
cd dgl 
export DGL_HOME=$(pwd)
bash script/build_dgl.sh -c
```

5. Finally, install the Python binding: 
```bash
cd python
python setup.py install
python setup.py build_ext --inplace
```

6. Install the compiled dgl package
```bash
uv pip install -e .
```

7. Go back to privmedrag repo root to install more dependencies: 
```bash
cd ..
uv pip install -e .
uv pip install numpy==1.26.4 pyyaml google-genai
```

## Data

### PrimeKG download

```bash
mkdir -p dataset/primekg/raw
wget -O dataset/primekg/raw/edges.csv https://dataverse.harvard.edu/api/access/datafile/6180616
wget -O dataset/primekg/raw/nodes.tsv https://dataverse.harvard.edu/api/access/datafile/6180617
```

### Preparing Patient EHR data

To prepare the data for synthetic MCQ generation:

```bash
mkdir -p dataset/synthea-dataset-100
wget -O dataset/synthea-dataset-100.zip https://github.com/lhs-open/synthetic-data/raw/main/record/synthea-dataset-100.zip
unzip dataset/synthea-dataset-100.zip -d dataset/synthea-dataset-100
python utils/preprocess_synthea_data.py
```

**To generate the synthetic data and run evaluations, please see: [./evaluate/README.md](./evaluate/README.md).**

## Acknowledgements

We build off the work of RGL as seen in [README_RGL.md](README_RGL.md).