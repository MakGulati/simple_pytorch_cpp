## Boilerplate Transfer Learning Example for running pytorch C++ distribution 
---
### Installing libtorch
At first, C++ distribution of pytorch needs to be installed mentioned in the [link](https://pytorch.org/cppdocs/installing.html).

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm -rf libtorch-shared-with-deps-latest.zip
```
### Building application using cmake
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
```

### Executing the compiled code from build directory
```bash
./simple_pytorch_cpp ../models/py_trace_model.pt 
```