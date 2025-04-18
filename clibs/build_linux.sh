# gcc -shared -o libretrieval.so -fPIC retrieve.c
# mv libretrieval.so ../rgl/graph_retrieval/

# g++ retrieve.cpp -o retrieve

c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) retrieve.cpp -o retrieval$(python3-config --extension-suffix)
mv *.so ../rgl/graph_retrieval/