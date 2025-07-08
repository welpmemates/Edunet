import matplotlib.pyplot as plt
import pandas as pd

# Load the dataframe
df_all_seasons = pd.read_pickle('df_all_seasons.pkl')

# Plotting kWh for each row (day) using index as x-axis
plt.figure(figsize=(14,6))
plt.bar(df_all_seasons.index, df_all_seasons['kwh'], color='orange')

plt.xlabel('Day Index')
plt.ylabel('Energy Output (kWh)')
plt.title('Day-wise Solar Panel Energy Output (Unaveraged)')
plt.tight_layout()
plt.show()