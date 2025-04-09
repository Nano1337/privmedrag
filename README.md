<!-- include logo svg in this markdown -->
<!-- <p align="center">
    <img src="rgl-logo.png" width="400"/>
</p> -->

# RGL - RAG-on-Graphs Library  

RGL is a **friendly and efficient Graph Retrieval-Augmented Generation (GraphRAG) library** for AI researchers, providing seamless integration with **DGL** and **PyG** while offering high-performance graph retrieval algorithms, many of which are optimized in **C++** for efficiency. 

## Features  
✅ **Seamless Integration** – Works smoothly with **DGL** and **PyG**  
⚡ **Optimized Performance** – C++-backed retrieval algorithms for speed  
🧠 **AI-Focused** – Tailored for **GraphRAG** research and applications  
🔗 **Scalability** – Handles large-scale graphs with ease  

## Homepage, Documentation and Paper

- Homepage: https://github.com/PyRGL/rgl
- Documentation: https://rgl.readthedocs.io
- Paper Access:
    - ArXiv: https://arxiv.org/abs/2503.19314

## Requirements

- DGL: https://www.dgl.ai/pages/start.html
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Installation

```bash
pip install rgl
```

## Compile C++ libraries

```bash
cd clibs
./build_linux.sh
```

## Run demos

```bash
cd demo
python demo_x.py
```

## Cite

```
@misc{li2025rglgraphcentricmodularframework,
      title={RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs}, 
      author={Yuan Li and Jun Hu and Jiaxin Jiang and Zemin Liu and Bryan Hooi and Bingsheng He},
      year={2025},
      eprint={2503.19314},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2503.19314}, 
}
```

## Setup

We manage dependencies using `uv` so please install it first then run the following command to install all dependencies.

```bash
uv sync
```

FAQ: 
- You might run into `dgl` dependency install issues and have to build from source on Apple Silicon:
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

To prepare to the data: 

```bash
wget -O dataset/synthea-dataset-100.zip https://github.com/lhs-open/synthetic-data/raw/main/record/synthea-dataset-100.zip
```

```bash
unzip dataset/synthea-dataset-100.zip -d dataset/synthea-dataset-100
```
