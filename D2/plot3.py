import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load y_test and y_pred
y_test = np.load('y_test.npy')
y_pred = np.load('y_pred.npy')

# 1. Scatter plot: Actual vs. Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual kWh")
plt.ylabel("Predicted kWh")
plt.title("Actual vs. Predicted kWh")

# Use min and max values for the diagonal line
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r--')  # Diagonal line red dash line

plt.grid(True)
plt.show()