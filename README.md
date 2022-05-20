# Schrödinger Approximation

## Building

(This program was tested in Linux with Clang 13.0.1 and GNU 12.1.0)

This will build everything
```bash
cmake . -B cmake-build
cmake --build cmake-build
```

This will build just the executable
```bash
cmake . -B cmake-build
cmake --build cmake-build --target SchroedingerApproximation
```

This will build the unit tests and the executable as a library
```bash
cmake . -B cmake-build
cmake --build cmake-build --target SchroedingerApproximation_TEST
```

This will build the executable as a library
```bash
cmake . -B cmake-build
cmake --build cmake-build --target SchroedingerApproximation_LIB
```

For debug version add the `-DCMAKE_BUILD_TYPE=Debug` flag to the first command

The resulting executable will then be in the __cmake-build__ directory