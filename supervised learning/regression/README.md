# Regression Learning Notes

This folder is a practical walkthrough of regression, starting from the simplest idea of fitting a line and moving toward regularized models such as Ridge, Lasso, and Elastic Net.

This README is meant to be a reference document you can come back to whenever something feels unclear.

## What Is Regression?

Regression is used when the target value is continuous.

Examples:
- Predicting house price
- Predicting salary
- Predicting package based on CGPA
- Predicting medical progression score

The basic idea is:

`input features -> model -> predicted number`

A regression model learns the relationship between the input variables and the target.

## What Is In This Folder?

This folder contains these notebooks:

| Notebook | Main idea |
| --- | --- |
| `simple_regression.ipynb` | One input feature, one output |
| `linear-regression.ipynb` | Linear regression with multiple features using a synthetic housing dataset |
| `multiple_linear_regression.ipynb` | Multiple linear regression with sklearn and from-scratch normal equation |
| `gradient_descent_simple_linear_regression.ipynb` | Learning `m` and `b` using gradient descent |
| `gradient_descent_multiple_linear_regression.ipynb` | Learning weights and bias for multiple features using gradient descent |
| `polynomial_linear_regression.ipynb` | Turning nonlinear curves into linear regression by feature expansion |
| `ridge_regression.ipynb` | L2 regularization |
| `lasso_regression.ipynb` | L1 regularization |
| `elastic_net_regression.ipynb` | Mix of Ridge and Lasso |

## Best Order To Study These Notebooks

1. `simple_regression.ipynb`
2. `linear-regression.ipynb`
3. `multiple_linear_regression.ipynb`
4. `gradient_descent_simple_linear_regression.ipynb`
5. `gradient_descent_multiple_linear_regression.ipynb`
6. `polynomial_linear_regression.ipynb`
7. `ridge_regression.ipynb`
8. `lasso_regression.ipynb`
9. `elastic_net_regression.ipynb`

This order helps because each step adds only one new idea.

## Common Notation

You will keep seeing these symbols:

- `x` = input feature
- `x1, x2, x3` = multiple features
- `y` = actual target value
- `y_hat` or `y^` = predicted target value
- `m` = slope in simple linear regression
- `b` = intercept
- `w` or `beta` = model coefficients
- `n` = number of training examples

## Core Evaluation Metrics

These are the most common metrics used across the notebooks:

### 1. Mean Absolute Error

`MAE = (1 / n) * sum(|y_i - y_hat_i|)`

This tells us the average absolute prediction error.

### 2. Mean Squared Error

`MSE = (1 / n) * sum((y_i - y_hat_i)^2)`

This penalizes large errors more strongly because of the square.

### 3. Root Mean Squared Error

`RMSE = sqrt(MSE)`

This brings the error back to the original target scale.

### 4. R-squared

`R^2 = 1 - (sum((y_i - y_hat_i)^2) / sum((y_i - y_mean)^2))`

Interpretation:
- `R^2 = 1` means perfect prediction
- `R^2 = 0` means no better than predicting the mean
- Negative `R^2` means worse than predicting the mean

## 1. Simple Linear Regression

Notebook: `simple_regression.ipynb`

### Goal

Predict one continuous target from one feature.

In your notebook:
- Input feature: `cgpa`
- Target: `package`

### Model Equation

`y = m*x + b`

Where:
- `m` is the slope
- `b` is the intercept

### What The Slope Means

If `m` increases, the line gets steeper.

If `m` is positive:
- as `x` increases, `y` tends to increase

If `m` is negative:
- as `x` increases, `y` tends to decrease

### Closed-Form Formula Used In The Scratch Class

The custom `MyLinearRegression` class computes:

`m = sum((x_i - x_mean) * (y_i - y_mean)) / sum((x_i - x_mean)^2)`

`b = y_mean - m * x_mean`

This is the direct mathematical solution for one-feature linear regression.

### Logic Of The Notebook

The notebook does two things:

1. It uses `sklearn.linear_model.LinearRegression`
2. It builds the same logic manually in a custom class

Flow:
- Load placement data
- Plot `cgpa` vs `package`
- Split into train and test
- Train a linear regression model
- Visualize the regression line
- Extract `m` and `b`
- Build a custom class that computes the same values manually

### Key Intuition

The model tries to draw the best straight line through the points so the prediction error is as small as possible.

### When To Use It

Use simple linear regression when:
- you have one main feature
- the relationship looks roughly linear
- you want the most interpretable starting model

## 2. Linear Regression With Multiple Features

Notebook: `linear-regression.ipynb`

### Goal

Predict a target using more than one input feature.

In your notebook:
- Features: `size`, `age`
- Target: `price`

### Model Equation

`y = b0 + b1*x1 + b2*x2 + ... + bn*xn`

For the housing example:

`price = b0 + b1*(size) + b2*(age)`

### Why It Is Still Called Linear Regression

It is called linear because the model is linear in the coefficients `b0, b1, b2, ...`.

Even with many features, the prediction is still a weighted sum plus intercept.

### Logic Of The Notebook

- Create synthetic housing data
- Plot the data
- Split into train and test
- Train `LinearRegression`
- Print intercept and coefficients
- Measure `MSE`, `RMSE`, and `R^2`
- Visualize a regression plane in 3D

### Interpretation Of Coefficients

If `b1 = 100` for `size`, then increasing house size by 1 unit increases predicted price by about 100, assuming other features stay fixed.

If `b2 = -3000` for `age`, then increasing age by 1 year decreases predicted price by about 3000, assuming other features stay fixed.

### Key Intuition

Each feature contributes separately to the final prediction.

The model learns how much each feature should push the prediction up or down.

## 3. Multiple Linear Regression From Scratch

Notebook: `multiple_linear_regression.ipynb`

This notebook is important because it shows both:
- a library implementation using sklearn
- a mathematical implementation from scratch

### Prediction Equation

`y_hat = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp`

### Matrix Form

`y_hat = X * beta`

If we add a column of ones to include the intercept, the full coefficient vector can be computed using the normal equation:

`beta = (X^T * X)^-1 * X^T * y`

### Why The Notebook Inserts A Column Of Ones

Your custom class does this:

`X_train = np.insert(X_train, 0, 1, axis=1)`

That first column of ones is used so the intercept becomes part of the matrix multiplication.

### Logic Of The Scratch Class

The class:
- adds the bias column
- computes `beta` using matrix algebra
- separates the first value as `intercept_`
- stores the remaining values as `coef_`
- predicts with:

`y_pred = X_test * coef + intercept`

### Important Condition

The normal equation needs `(X^T * X)` to be invertible.

If features are highly correlated, this matrix can become unstable or singular.
That is one reason regularized models like Ridge and Elastic Net become useful later.

### Key Intuition

This notebook shows that linear regression is not magic.
It is just linear algebra applied to minimize squared error.

## 4. Gradient Descent On Simple Linear Regression

Notebook: `gradient_descent_simple_linear_regression.ipynb`

### Why This Notebook Matters

Earlier notebooks show direct formulas.
This notebook shows how the model can learn step by step instead of solving everything in one shot.

### Cost Function

The notebook uses Mean Squared Error:

`J(m, b) = (1 / n) * sum((y_i - (m*x_i + b))^2)`

### Gradient Equations

`dm = (-2 / n) * sum(x_i * (y_i - y_hat_i))`

`db = (-2 / n) * sum(y_i - y_hat_i)`

### Update Rules

`m = m - learning_rate * dm`

`b = b - learning_rate * db`

### Logic Of The Notebook

- Start with `m = 0` and `b = 0`
- Predict values
- Measure error
- Compute the gradients
- Update `m` and `b`
- Repeat for many epochs
- Track cost history

### What Gradient Descent Is Doing

Think of the cost function as a hill surface.
Gradient descent keeps moving in the direction that reduces the error fastest.

### Why Learning Rate Matters

If learning rate is too small:
- training is very slow

If learning rate is too large:
- the model may overshoot the best values
- loss may oscillate or diverge

## 5. Gradient Descent On Multiple Linear Regression

Notebook: `gradient_descent_multiple_linear_regression.ipynb`

### Prediction Equation

`y_hat = X*w + b`

Where:
- `w` is the vector of weights
- `b` is the bias

### Cost Function

`J(w, b) = (1 / n) * sum((y_i - y_hat_i)^2)`

### Gradient Equations

`dw = (2 / n) * X^T * (y_hat - y)`

`db = (2 / n) * sum(y_hat - y)`

### Update Rules

`w = w - learning_rate * dw`

`b = b - learning_rate * db`

### Logic Of The Notebook

- Start with zero weights
- Compute predictions with `np.dot(X, weights) + bias`
- Measure the errors
- Compute gradients for all features together
- Update weights and bias
- Save cost history for visualization

### Key Intuition

This is the same idea as simple gradient descent, but now each feature gets its own weight.

## 6. Polynomial Regression

Notebook: `polynomial_linear_regression.ipynb`

### Main Idea

Sometimes the relationship between `x` and `y` is curved, not straight.

A straight line underfits curved data.

Polynomial regression solves this by creating extra features such as:

- `x`
- `x^2`
- `x^3`

### Example Equation

For degree 2:

`y = beta0 + beta1*x + beta2*x^2`

### Very Important Concept

Polynomial regression is still linear regression.

It is still linear in the coefficients:
- `beta0`
- `beta1`
- `beta2`

The curve comes from the transformed features, not because the learning algorithm changed into something completely different.

### Logic Of The Notebook

- Generate synthetic nonlinear data
- Fit normal linear regression first
- Observe weak fit
- Create polynomial features using `PolynomialFeatures(degree=2)`
- Train linear regression on transformed features
- Evaluate with `R^2`

### Why The Fit Improves

The transformed feature matrix can represent curved patterns, so the model is flexible enough to fit data that a straight line cannot explain well.

### Risk

Higher-degree polynomials can overfit.

If degree becomes too high:
- training performance may look great
- test/generalization performance may get worse

## 7. Ridge Regression

Notebook: `ridge_regression.ipynb`

### Problem Ridge Solves

In ordinary linear regression, very large coefficients can appear when:
- features are correlated
- data is noisy
- the model becomes too sensitive

Ridge adds an L2 penalty to discourage large weights.

### Objective Function

`Loss = sum((y_i - y_hat_i)^2) + alpha * sum(w_j^2)`

### Meaning Of The Penalty

The model is now trying to do two things at once:
- fit the data well
- keep coefficients small

### Key Property

Ridge usually shrinks coefficients toward zero, but usually not exactly to zero.

That means:
- it reduces variance
- it improves stability
- it usually keeps all features in the model

### Logic Of Your Notebook

The notebook:
- creates a synthetic housing dataset
- standardizes features
- centers the target
- runs coordinate descent
- recovers coefficients on the original scale
- compares how coefficients shrink for different `alpha` values

### Why Standardization Matters Here

Regularization depends on coefficient magnitude.

If features are on very different scales, the penalty becomes unfair.
Standardization makes the penalty more meaningful.

### Coordinate Descent Update Used

For Ridge, the notebook updates each coefficient as:

`w_j = rho_j / (z_j + alpha)`

Where:
- `rho_j` is the partial residual correlation for feature `j`
- `z_j` is the sum of squared values of feature `j`

### When To Prefer Ridge

Use Ridge when:
- most features seem useful
- features are correlated
- you want smoother, more stable coefficients

## 8. Lasso Regression

Notebook: `lasso_regression.ipynb`

### Problem Lasso Solves

Like Ridge, Lasso controls coefficient size.
But it uses a different penalty:

`Loss = sum((y_i - y_hat_i)^2) + alpha * sum(|w_j|)`

This is the L1 penalty.

### Key Property

Lasso can push some coefficients exactly to zero.

That means it can perform feature selection automatically.

### Soft Thresholding

Your notebook uses the soft-threshold function:

`S(value, penalty)`

It behaves like this:
- if value is larger than the penalty, shrink it down
- if value is smaller than negative penalty, shrink it up
- if value is near zero, force it to zero

In equation form:

`S(rho, alpha) = rho - alpha, if rho > alpha`

`S(rho, alpha) = rho + alpha, if rho < -alpha`

`S(rho, alpha) = 0, otherwise`

### Coordinate Descent Update Used

`w_j = S(rho_j, alpha) / z_j`

### Logic Of The Notebook

- build synthetic data
- standardize features
- center target
- run coordinate descent with soft thresholding
- compare coefficients for different `alpha`
- show that some weak coefficients may become exactly zero

### When To Prefer Lasso

Use Lasso when:
- you think some features are not useful
- you want a simpler model
- feature selection is important

### Limitation

When features are strongly correlated, Lasso may choose one and suppress another in a less stable way.

## 9. Elastic Net Regression

Notebook: `elastic_net_regression.ipynb`

### Why Elastic Net Exists

Ridge is stable with correlated features.
Lasso can select features.

Elastic Net combines both ideas.

### Objective Function

`Loss = sum((y_i - y_hat_i)^2) + alpha * [l1_ratio * sum(|w_j|) + (1 - l1_ratio) * sum(w_j^2)]`

### Meaning Of `l1_ratio`

- `l1_ratio = 0.0` behaves like Ridge
- `l1_ratio = 1.0` behaves like Lasso
- values in between mix both penalties

### Why It Is Useful

Elastic Net is often a strong default when:
- features are correlated
- you want some sparsity
- pure Lasso is too unstable
- pure Ridge keeps too many weak features

### Logic Of Your Notebook

The notebook:
- creates a synthetic housing dataset
- intentionally includes correlated features such as `size_sqft` and `garage_sqft`
- standardizes features
- centers the target
- applies coordinate descent
- separates the L1 part and L2 part of the penalty
- compares different values of `alpha` and `l1_ratio`
- predicts the price of a new house

### Penalties Used In The Notebook

`l1_penalty = alpha * l1_ratio`

`l2_penalty = alpha * (1 - l1_ratio)`

### Coordinate Descent Update Used

`w_j = S(rho_j, l1_penalty) / (z_j + l2_penalty)`

This is exactly why Elastic Net sits between Lasso and Ridge:
- soft-thresholding comes from the L1 part
- the denominator shrinkage comes from the L2 part

### Practical Meaning

- Larger `alpha` means stronger regularization
- Larger `l1_ratio` means more Lasso-like behavior
- Smaller `l1_ratio` means more Ridge-like behavior

## Quick Comparison Of The Models

| Model | Formula shape | Main strength | Main risk |
| --- | --- | --- | --- |
| Simple Linear Regression | `y = m*x + b` | Very easy to understand | Only one feature |
| Multiple Linear Regression | `y = b0 + b1*x1 + ... + bp*xp` | Uses many features | Sensitive to multicollinearity |
| Polynomial Regression | `y = b0 + b1*x + b2*x^2 + ...` | Fits curved trends | Can overfit |
| Ridge | MSE + L2 penalty | Stable with correlated features | Does not remove features exactly |
| Lasso | MSE + L1 penalty | Feature selection | Can be unstable with correlated features |
| Elastic Net | MSE + L1 + L2 | Good balance of stability and sparsity | Needs tuning of two hyperparameters |

## Important Concepts That Connect All Notebooks

### 1. Prediction Function

Every model is basically trying to learn:

`prediction = weighted combination of features + intercept`

The differences come from:
- how features are represented
- how coefficients are learned
- whether penalties are added

### 2. Loss Function

Training means minimizing a loss function.

For most notebooks here, the base loss is squared error:

`sum((y_i - y_hat_i)^2)`

Regularized models add extra penalty terms.

### 3. Bias And Variance

Very simple models may underfit.
Very flexible models may overfit.

Regularization helps control overfitting by shrinking coefficients.

### 4. Feature Scaling

Feature scaling becomes especially important when:
- using gradient descent
- using regularization

If one feature is much larger than another, optimization and penalties can behave badly.

### 5. Why Polynomial Regression Is Still Regression

Because the model is still linear in the coefficients.
Only the input representation changes.

## Common Doubts And Quick Answers

### "Why is it called linear if there are many features?"

Because the model is linear in the coefficients.
Multiple features do not make it nonlinear.

### "Why is polynomial regression still called linear regression?"

Because the coefficients are still combined linearly.
Only the features are transformed.

### "Why do we split train and test data?"

To check whether the model generalizes to unseen data instead of only memorizing the training set.

### "What is the difference between coefficient and intercept?"

- Intercept = baseline prediction when all feature values are zero
- Coefficient = how much prediction changes when a feature changes by one unit, holding others fixed

### "Why do Ridge and Lasso need standardization?"

Because the penalty acts on coefficient size.
Without scaling, features with large numeric ranges can distort the regularization effect.

### "When should I use gradient descent instead of the normal equation?"

Use gradient descent when:
- data is large
- the direct matrix inverse is expensive
- you want an iterative optimization view

Use the normal equation when:
- the dataset is small to medium
- you want a direct closed-form solution

### "What does alpha do?"

`alpha` controls regularization strength.

Larger `alpha`:
- more shrinkage
- simpler model
- possibly higher bias

Smaller `alpha`:
- less shrinkage
- more flexible model
- possibly higher variance

### "What does l1_ratio do in Elastic Net?"

It controls the mix between Lasso and Ridge.

- closer to `1` means more Lasso-like
- closer to `0` means more Ridge-like

## A Simple Mental Map

If you ever feel lost, remember this path:

1. Start with a line
2. Add more features
3. Learn coefficients by direct math
4. Learn coefficients by gradient descent
5. Add polynomial features for curves
6. Add regularization to control overfitting
7. Mix L1 and L2 when needed

That is the whole story of this folder.

## How To Revisit This Folder Later

If you want a quick refresh:

### For fundamentals

Read:
- `simple_regression.ipynb`
- `linear-regression.ipynb`

### For the actual math

Read:
- `multiple_linear_regression.ipynb`
- `gradient_descent_simple_linear_regression.ipynb`
- `gradient_descent_multiple_linear_regression.ipynb`

### For nonlinear relationships

Read:
- `polynomial_linear_regression.ipynb`

### For overfitting control and regularization

Read:
- `ridge_regression.ipynb`
- `lasso_regression.ipynb`
- `elastic_net_regression.ipynb`

## Final Summary

All notebooks in this folder are teaching the same central idea:

Find a rule that maps features to a continuous target while minimizing prediction error.

The progression is:
- simple line
- multiple features
- optimization
- nonlinear feature engineering
- regularization

If you understand these five things, the whole folder will start to feel connected:

1. prediction equation
2. loss function
3. optimization method
4. feature representation
5. regularization

Whenever you get stuck, come back to these five anchors first.
