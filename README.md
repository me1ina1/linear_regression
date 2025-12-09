# Linear Regression - Implementation and Analysis

## Overview
This project demonstrates the implementation of linear regression with gradient descent in Python. It includes both theoretical foundations and practical applications on the Iris dataset.

## Project Structure

### 1. Basic Implementation
- **Linear Function**: `f(m, x, t) = m*x + t`
- **Error Calculation**: Mean Squared Error (MSE)
- **Gradient Descent**: Optimization of parameters m and t

### 2. Data Generation
Two data sources are used:
- **Synthetic Data**: Randomly generated points with controlled noise
- **Iris Dataset**: Real botanical data (Setosa species)

### 3. Visualizations

#### Graph 1: Error Surface
<img width="614" height="418" alt="image" src="https://github.com/user-attachments/assets/05452cfa-d3c4-40b6-8211-b9ce10917022" style="margin-left: 0; display: block;"/>



This visualization shows:
- The convexity of the error function
- The global minimum at m ≈ 0.1675
- The error rate of ~0.01 at the optimal point

**Interpretation**: The smooth, convex shape guarantees that gradient descent converges to the global minimum.

#### Graph 2: Gradient Descent Result
<img width="608" height="412" alt="image" src="https://github.com/user-attachments/assets/d42346a9-8d03-4c53-af18-4b8b24014ea2" style="margin-left: 0; display: block;"/>


This visualization shows:
- Original data as red points
- Fitted regression line in blue
- Error reduction from ~12956 to ~35 after 1000 epochs

**Interpretation**: The algorithm successfully finds the best linear approximation of the data.

## Technical Details

### Hyperparameters
```python
Learning Rate (L) = 0.0001
Epochs (E) = 1000
```

### Gradient Formulas
```
Dm = (-2/n) * Σ[x * (y - (m*x + t))]
Dt = (-2/n) * Σ[y - (m*x + t)]
```

## Results

### Iris Dataset (Setosa)
- **Features**: Petal Length vs. Petal Width
- **Initial Error**: ~0.01
- **Optimal Parameters**: m ≈ 0.1675, t ≈ 0

### Synthetic Data
- **Initial Error**: 12956.45
- **Final Error**: 35.25 (99.7% reduction)
- **Convergence**: Stable after ~500 epochs

## Usage

```python
# Load data
iris = load_iris("iris_daten.csv", 100)
data = [(x, y) for x, y in zip(
    iris.get_index(2, "setosa"), 
    iris.get_index(3, "setosa")
)]

# Train model
m, t = 0, 0
for i in range(1000):
    Dm = (-2/len(data)) * sum([x * (y - (m*x+t)) for x, y in data])
    Dt = (-2/len(data)) * sum([y - f(m, x, t) for x, y in data])
    m -= 0.0001 * Dm
    t -= 0.0001 * Dt
```

## Dependencies
```python
numpy
matplotlib
csv (standard library)
random (standard library)
math (standard library)
```

## Conclusions

1. **Convergence**: Gradient descent reliably converges to the optimal solution
2. **Scalability**: The method works for both synthetic and real data
3. **Visualization**: Error reduction is clearly visible and quantifiable

