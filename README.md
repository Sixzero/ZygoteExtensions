# ZygoteExtensions
Some functions to extend Zygote. 
It augment gradient operations.

Everything is in one file [ZygoteExtensions.jl](https://github.com/Cvikli/ZygoteExtensions.jl/blob/main/src/ZygoteExtensions.jl). Extremly hard to read I know... 

# ChatGPT
## Module Contents
This module extends the capabilities of Zygote, a popular automatic differentiation library in Julia, particularly focusing on machine learning applications. Key components include:

- **Vector Normalization:** `vnorm` function for normalizing vectors.
- **Non-Zero Averaging:** `mean_nonzero` function to compute mean values, excluding zeros.
- **Softmax Overloads:** Enhanced `softmax` functions for different data structures.
- **Gradient Timing:** `grad_timer` and `@gtime` macro for measuring performance of gradient computations.
- **Observation Utilities:** `observe` function and its adjoint for monitoring variables during the differentiation process.
- **Array Manipulations:** Various utility functions like `antizero`, `onehot`, `get_slice`, `assign_eles!`, `lax_scan`, and their respective gradient functions.
- **Custom Assertions:** Integration with `ToggleableAsserts` for conditional assertion checks.
- **Dimension Handling:** Enhanced dimension management with `Boilerplate` utilities.

## Usage
Integrate this module into your Julia machine learning projects to leverage advanced gradient operations, performance monitoring, and array manipulation utilities, enhancing both the efficiency and effectiveness of your model training and evaluation processes.
