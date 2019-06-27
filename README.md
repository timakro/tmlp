This is the [Teeworlds](https://teeworlds.com) game server modification for the Teeworlds Machine Learning Project.

## Building from source

The software was only tested on Linux. Make sure you have all the dependencies from the [building instructions](https://github.com/teeworlds/teeworlds#building-on-linux-or-macos) of the vanilla Teeworlds server.

There are x86-64 shared objects for the TensorFlow C++ API in this repository. You can find them in `other/tensorflow`. If you are on a different architecture or want CUDA support you should compile them yourself and put them in that directory.

## Building TensorFlow C++ API

Also check out the [official building instructions](https://www.tensorflow.org/install/source). Clone the [TensorFlow git repository](https://github.com/tensorflow/tensorflow) and install [Bazel](https://www.bazel.build/). For Arch Linux there is the [bazel](https://www.archlinux.org/packages/?name=bazel) package.

Cd into the the repository and checkout a TensorFlow 2.0 Beta version. The software was developed and tested with `v2.0.0-beta1`. Before starting Bazel run `./configure`. Possible options to pass to Bazel are listed in the configure output.

    bazel build --config=opt //tensorflow:libtensorflow_cc.so

## Building the server

Create a `build` folder in this repository and cd into it.

    cmake -DCLIENT=OFF ..
    make teeworlds_srv
