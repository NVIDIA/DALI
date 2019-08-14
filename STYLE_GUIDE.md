# DALI Coding Style Guide

This document describes DALI Coding Style Guide. Rules specified here take precedence
over [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) which should
be followed in the remaining cases.

The code should always pass the current `make lint` check.

## Changes compared to Google C++ Style Guide

Google C++ Style Guide is the default *style* guide. In places where it limits use of common
C++ idioms or language features it is discouarged.

### C++ Version

DALI uses C++14 standard as this is the most recent version supported with CUDA.

### Line length

We use line length limit eqaul 100.

### Reference arguments

Parameters can be passed as non-const lvalue reference. [Google rule](https://google.github.io/styleguide/cppguide.html#Reference_Arguments)
prohibits semantically valid restriction of not passing null pointer
and introduces ugly code like `foo(&bar)` or `(*buf)[i]`.

## DALI specific rules

### DALI Kernels argument order

DALI Kernels follow order of Outputs, Inputs, Arguments - where Output and Inputs are
expected to be views to Tensors (TensorLists) and Arguments are other inputs.

The same order should be maintained for Kernel template arguments.
See [the example](dali/kernels/kernel.h) kernel implementation for details.

### Documentation

DALI uses Doxygen for C++ code documentation with Javadoc-styled comments:

```
/**
 * ... text ...
 */
```


## Unspecified cases

When the style is left unspecified please follow the one used most in the current codebase.