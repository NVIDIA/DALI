# DALI Coding Style Guide

This document describes DALI Coding Style Guide.

DALI's C++ code follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
with the exceptions detailed below.

DALI's Python code uses [Black](https://github.com/psf/black) for formatting, with the configuration
specified in the `pyproject.toml` file, and is checked with [flake8](https://github.com/PyCQA/flake8)
linter according to the `.flake8` configuration file.

The code should always pass the current `make lint` check, where C++ code and Python formatting
are verified.

## Changes compared to Google C++ Style Guide

Google C++ Style Guide is the default *style* guide. In places where it limits use of common
C++ idioms or language features it is discouraged.

### C++ Version

DALI uses C++17 standard as this is the most recent version supported with CUDA versions used
in the project.

### Line length

We use a line length limit equal to 100.

### Reference arguments

Parameters can be passed as non-const lvalue reference. [Google rule](https://google.github.io/styleguide/cppguide.html#Reference_Arguments)
prohibits semantically valid restriction of not passing null pointer
and introduces ugly code like `foo(&bar)` or `(*buf)[i]`.

### Test suites naming guide

We use GTest for most of testing code in DALI. Names of TestSuites should start with a capital letter and end with `Test`.
Additionally, both suite and case name mustn't contain underscores (`_`).
For details on the latter, cf. [GTest FAQ](https://github.com/google/googletest/blob/master/googletest/docs/faq.md#why-should-test-suite-names-and-test-names-not-contain-underscore).
Examples:
```
TEST(MatTest, IdentityMatrix) {}  // OK
TEST_F(PregnancyTest, AlwaysPositive) {}  // OK
TYPED_TEST(CannyOperatorTest, EmptyImage) {}  // OK
TYPED_TEST_SUITE(Skittles, InTheSky);  // Wrong! Should be "SkittlesTest"
INSTANTIATE_TYPED_TEST_SUITE_P(Integral, HelloTest, IntegralTypes);  // OK. "Integral" is a prefix for type-parameterized test suite

```


## DALI specific rules

### DALI Kernels argument order

DALI Kernels follow order of Outputs, Inputs, Arguments - where Output and Inputs are
expected to be views to Tensors (TensorLists) and Arguments are other inputs.

The same order should be maintained for Kernel template arguments.
See [the example](dali/kernels/kernel.h) kernel implementation for details.

The order of the arguments is following memcpy semantics.

### Documentation

DALI uses Doxygen for C++ code documentation with Javadoc-styled comments:

```
/**
 * ... text ...
 */
```


## Unspecified cases

When the style is left unspecified please follow the one used most in the current codebase.
If there is no precedence in the codebase, we are open to discussion, but we hold the final
word to avoid endless discussion in that matter.
