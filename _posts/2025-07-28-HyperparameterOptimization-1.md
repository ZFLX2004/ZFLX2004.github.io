---
layout: post
title: Hyperparameter Optimization - 1
subtitle: A Brief Introduction to HPO
cover-img: ../assets/img/HPO/cover-background.png
thumbnail-img: ../assets/img/HPO/cover.png
share-img: ../assets/img/HPO/cover.png
tags: [python, AI, model optimization]
author: 吴耀廷 Yaoting Wu
---

In machine learning, the performance of a model, like accuracy, generalization ability etc. is often affected by its **parameter settings**. For example, consider that you are building one Transformer to predict one time series data, and you may wonder how to determine some parameters like `d_models` and `n_heads`. If you make it large, the training time and cost may increase, and a risk of overfitting may exist. On the contrary, if you make it small, the accuracy may be greatly reduced. This is the topic of this article.

## 1. Understanding Parameters and Hyperparameters

### 1.1. Parameters and Hyperparameters

In any machine learning model, there are two distinct kinds of parameters:

1.  **Model Parameters**: These are internal to the model and their values are **learned from the data** during the training process. You don't set them manually.
    -   **Examples**: The weights ($w$) and biases ($b$) in a neural network, or the coefficients in a linear regression model. These are the values the model "learns" to make accurate predictions.

2.  **Hyperparameters**: These are external configuration settings for the model and the training algorithm. They are **set by the practitioner *before* training begins**. They control the overall behavior, structure, and performance of the model.
    -   **Examples**: The learning rate, the number of layers in a deep neural network, the `d_model` (embedding dimension) and `n_heads` (number of attention heads) in a Transformer.

The core challenge is that there's no magic formula to determine the best hyperparameters. Their optimal values depend on the specific dataset and task. This is why we need **Hyperparameter Optimization (HPO)**.

### 1.2. What is Hyperparameter Optimization?

Hyperparameter Optimization (HPO) is the process of **automatically searching for the optimal set of hyperparameters** for a machine learning model to achieve the best performance on a given task.

- **Goal**: To maximize a performance metric (e.g., accuracy, F1-score) or minimize a loss function on a validation dataset.
- **Process**: It involves defining a search space of possible hyperparameter values and using a systematic method to explore this space and find the combination that yields the best result.

## 2. Understanding the Hyperparameter Space

The **Hyperparameter Space** (or Search Space) is the set of all possible hyperparameter combinations that we want to explore. Defining this space is the first step in any HPO task. A well-defined space is crucial for finding the best model efficiently.

The space is defined by its:

1.  **Dimensions**: Each hyperparameter you decide to tune is a dimension in this space. A Transformer model might have dimensions like `learning_rate`, `d_model`, `n_heads`, and `num_layers`.

2.  **Type of Hyperparameters**:
    -   **Continuous**: Can take any real value within a range (e.g., `learning_rate` from `0.0001` to `0.1`).
    -   **Integer**: Can take any integer value within a range (e.g., `num_layers` from `2` to `8`).
    -   **Categorical**: Can take one value from a discrete set of choices (e.g., `activation_function` from `['relu', 'tanh', 'gelu']`).

3.  **Structure and Constraints**:
    -   The space isn't always a simple grid. Some hyperparameters can be **conditional**. For example, the hyperparameters for an Adam optimizer (like `beta1`, `beta2`) are only relevant if you choose `'Adam'` as your `optimizer_type`.
    -   There can also be **constraints**. For instance, in a Transformer, `d_model` must be divisible by `n_heads`.

### Example: Defining a Search Space for a Transformer

Let's imagine we are tuning a Transformer model. Our search space might look like this:

| Hyperparameter       | Type        | Range / Choices                               | Description                               |
|----------------------|-------------|-----------------------------------------------|-------------------------------------------|
| `learning_rate`      | Continuous  | Log-uniform between `1e-5` and `1e-2`         | Controls how much the model weights are updated. |
| `d_model`            | Integer     | `[128, 256, 512]`                             | The dimensionality of the input embeddings. |
| `n_heads`            | Integer     | `[4, 8, 16]`                                  | Number of attention heads.                |
| `num_encoder_layers` | Integer     | `[2, 3, 4, 5, 6]`                             | Number of layers in the encoder stack.    |
| `dropout`            | Continuous  | Uniform between `0.1` and `0.3`               | Regularization to prevent overfitting.    |
| `activation`         | Categorical | `['relu', 'gelu']`                            | The activation function in feed-forward layers. |

**Constraint**: `d_model` must be divisible by `n_heads`. An HPO algorithm must respect this constraint when sampling combinations.

The goal of HPO is to navigate this multi-dimensional, complex space to find the point that corresponds to the best-performing model.

## 3. Common Hyperparameter Optimization Methods

| Category                     | Methods                                          |
|------------------------------|--------------------------------------------------|
| **Manual / Basic**           | Grid Search, Random Search                       |
| **Sequential / Adaptive**    | Bayesian Optimization, Hyperband, Successive Halving |
| **Gradient-based**           | Gradient-Based Optimization                      |
| **Evolutionary / Metaheuristics** | Genetic Algorithms, Particle Swarm Optimization |
| **Ensemble / Hybrid**        | BOHB (Bayesian + Hyperband), Optuna, SMAC        |

### 1. Grid Search

**Idea**: Try all possible combinations in a discrete grid of hyperparameters.

- **Pros**: Simple and exhaustive.
- **Cons**: Computationally expensive; suffers from the "curse of dimensionality".

### 2. Random Search

**Idea**: Randomly sample combinations of hyperparameters.

- **Pros**: More efficient than grid search; can explore wide space quickly.  
- **Cons**: Still inefficient in high-dimensional spaces.

### 3. Bayesian Optimization (BO)

**Idea**: Build a probabilistic surrogate model (e.g., Gaussian Process, Tree Parzen Estimator) to model the objective function, and use it to select promising configurations.

- **Acquisition Functions**: Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), etc.  
- **Pros**: Efficient in low-dimensional hyperparameter spaces.  
- **Cons**: Can be slow with high-dimensional or discrete spaces.

### 4. Hyperband

**Idea**: Allocate resources (e.g., training epochs) adaptively using *Successive Halving*, evaluating many configurations with small budgets and keeping only the top performers.

- **Key Concepts**:  
  - Quickly explores many configurations with low budget  
  - Successively halves the set of configurations based on performance  
- **Pros**: Very efficient for deep learning with early-stopping.  
- **Cons**: Assumes that early performance correlates with final performance.

### 5. BOHB (Bayesian Optimization + Hyperband)

**Idea**: Combines Bayesian Optimization’s smart sampling with Hyperband’s resource allocation.

- Bayesian model proposes new configurations.  
- Hyperband allocates budgets and prunes unpromising runs.

### 6. Gradient-Based Optimization

**Idea**: When the validation metric is differentiable w.r.t. hyperparameters, use gradient descent to update them directly.

- **Pros**: Efficient when applicable.  
- **Cons**: Rarely practical—most hyperparameters are non-differentiable.

### 7. Evolutionary Algorithms / Metaheuristics

**Idea**: Use biologically inspired operations—mutation, crossover, and selection—to evolve populations of hyperparameter sets.

- **Algorithms**: Genetic Algorithms (GA), Particle Swarm Optimization (PSO), Differential Evolution (DE)  
- **Pros**: Good for discrete, non-smooth, or noisy spaces.  
- **Cons**: May require many function evaluations.

### 8. Reinforcement Learning-based Methods

**Idea**: Train an RL agent to generate hyperparameter configurations, often used in Neural Architecture Search (NAS).

- **Pros**: Can explore complex, structured search spaces.  
- **Cons**: Extremely resource-intensive.

## 4. Conclusion

Hyperparameter optimization is a critical step in building high-performance machine learning models. While manual tuning can be intuitive, it is often time-consuming and suboptimal. Automated HPO methods provide a systematic way to navigate the complex hyperparameter space and discover configurations that unlock a model's true potential.

This article has provided an overview of the landscape of HPO, from simple methods like Grid and Random Search to more sophisticated, adaptive strategies like Bayesian Optimization and Hyperband. The key takeaway is that there is no one-size-fits-all solution. The choice of an HPO method depends on several factors:

-   **Computational Budget**: How much time and resources can you afford? Hyperband and BOHB are excellent for limited budgets.
-   **Dimensionality of the Space**: How many hyperparameters are you tuning? Bayesian Optimization shines in low-dimensional spaces, while Random Search can be a surprisingly effective baseline for high-dimensional ones.
-   **Nature of the Hyperparameters**: Are they continuous, discrete, or conditional? Some frameworks handle complex, conditional spaces better than others.22

As models become more complex, intelligent HPO techniques are no longer a luxury but a necessity. By leveraging these methods, practitioners can save valuable time, improve model performance, and gain deeper insights into their models' behavior. In the next part of this series, we will dive into a practical example, using a popular library to optimize a real-world model.

### Summary Table

| Method             | Search Type               | Suitable For          | Efficiency      |
|--------------------|---------------------------|-----------------------|-----------------|
| Grid Search        | Exhaustive                | Small spaces          | ❌ Low          |
| Random Search      | Randomized                | Any space             | ✅ Medium       |
| Bayesian Opt.      | Model-based sequential    | Low-D spaces          | ✅ High         |
| Hyperband          | Bandit-style              | Deep learning models  | ✅✅ High        |
| BOHB               | Hybrid                    | All spaces            | ✅✅✅ Very High  |
| Evolutionary (GA)  | Population-based          | Non-smooth spaces     | ✅ Medium       |
| Gradient-based     | Differentiable hyperparams| Special use-cases     | ✅ High         |
| RL-based (NAS)     | Policy search             | Architecture search   | ✅❌ Very High   |
