This is the [Teeworlds](https://teeworlds.com) game server modification for the Teeworlds Machine Learning Project.

## Building from source

The software was only tested on Linux. Make sure you have all the dependencies from the [building instructions](https://github.com/teeworlds/teeworlds#building-on-linux-or-macos) of the vanilla Teeworlds server.

There are x86-64 shared objects for the TensorFlow C++ API in this repository. You can find them in `other/tensorflow`. If you are on a different architecture or want CUDA support you should compile them yourself and put them in that directory.

## Building TensorFlow C++ API

Also check out the [official building instructions](https://www.tensorflow.org/install/source). Clone the [TensorFlow git repository](https://github.com/tensorflow/tensorflow) and install [Bazel](https://www.bazel.build/). For Arch Linux there is the [bazel](https://www.archlinux.org/packages/?name=bazel) package.

Cd into the the repository and checkout a TensorFlow 1.x version. The software was developed and tested with `v1.14.0`.

    ./configure
    bazel build --config=opt //tensorflow:libtensorflow_cc.so
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 //tensorflow:libtensorflow_cc.so

## Building the server

Create a `build` folder in this repository and cd into it.

    cmake -DCMAKE_BUILD_TYPE=Release -DCLIENT=OFF ..
    make teeworlds_srv
