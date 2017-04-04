## NeuralNetwork

This repository contains code samples for my multilayer perceptron network  in C++.

Feel free to modify the code and bug reports are welcome.

### How to build

+ Setup Eigen3 library: sudo apt-get install libeigen3-dev

```
mkdir build
cd build
cmake -DEXT_INCS="./MyLib;/usr/include/eigen3" -DEXT_LIBS=-L"./MyLib/buildLinux/lib -lMyLib" -DCMAKE_BUILD_TYPE=Release ..
```

### License
It is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php).
