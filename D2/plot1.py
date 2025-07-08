import matplotlib.pyplot as plt
import pandas as pd

# Load the dataframe
df_all_seasons = pd.read_pickle('df_all_seasons.pkl')

# Boxplot of kWh by season
df_all_seasons.boxplot(column='kwh', by='season', grid=False, figsize=(8,6))
plt.title('Solar Panel Energy Output (kWh) by Season')
plt.suptitle('')  # Remove automatic subtitle
plt.xlabel('Season')
plt.ylabel('Energy Output (kWh)')
plt.show()