# Machine Learning Assignment-1 README

## Usage

1. **Open your Jupyter notebook**
2. **Import libraries**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    ```
3. **Read CSV Files**
    ```python
    # Example: X_data = pd.read_csv('linearX.csv')
    # Example: Y_data = pd.read_csv('linearY.csv')
    ```
4. **Partial Derivative of Cost Function**
    <img src="drv_ML.png" width="300px">

5. **Use in Model Building**
    ```python
    X = (X - X.mean()) / (X.std())

    # Your model building code here
    m = 0
    c = 0
    L = 0.5
    epochs = 1000
    n = float(len(X))

    for i in range(epochs):
        Y_pred = m * X + c
        cost = (1/n) * sum((Y - Y_pred)**2)
        print(f'Epoch {i+1}, Cost: {cost}')
        D_m = (-2/n) * sum(X * (Y - Y_pred))
        D_c = (-2/n) * sum(Y - Y_pred)
        m = m - L * D_m
        c = c - L * D_c

        print("Slope (m):", m)
        print("Intercept (c):", c)
    ```
    Ensure you adjust the code according to your specific model-building requirements.
    
6. **Standardize Input**
    ```python
    X = (X - X.mean()) / (X.std())
    ```
    ⚠️ **Important:** Standardize your input to prevent explosions!
    
7. **Answer the Following Questions**

## Additional Tips

- Make sure to explore and visualize your data using `matplotlib`.
- Experiment with different algorithms for model building.
- Document your code for better understanding.


