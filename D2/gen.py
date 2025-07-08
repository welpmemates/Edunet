import numpy as np
import pandas as pd

feature_ranges = {
    'summer': {
        'irradiance': (600, 1000),
        'humidity': (10, 50),
        'wind_speed': (0, 5),
        'ambient_temperature': (30, 45),
        'tilt_angle': (10, 40),
    },
    'winter': {
        'irradiance': (300, 700),
        'humidity': (30, 70),
        'wind_speed': (1, 6),
        'ambient_temperature': (5, 20),
        'tilt_angle': (10, 40),
    },
    'monsoon': {
        'irradiance': (100, 600),
        'humidity': (70, 100),
        'wind_speed': (2, 8),
        'ambient_temperature': (20, 35),
        'tilt_angle': (10, 40),
    }
}

# Summer months with exact days
summer_months_days = {
    'March': 31,
    'April': 30,
    'May': 31,
    'June': 30
}

# Winter months with exact days
winter_months_days = {
    'November': 30,
    'December': 31,
    'January': 31,
    'February': 28  # Not considering leap year here; can be adjusted if needed
}

# Monsoon months with exact days
monsoon_months_days = {
    'July': 31,
    'August': 31,
    'September': 30,
    'October': 31
}

def calculate_kwh(season, irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    if season == 'summer':
        return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))
    elif season == 'winter':
        return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))
    elif season == 'monsoon':
        return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))
    else:
        raise ValueError(f"Unknown season: {season}")

def generate_season_data_by_month(season, feature_ranges, months_days):
    data = []
    for month, days in months_days.items():
        for _ in range(days):
            irr = np.random.uniform(*feature_ranges[season]['irradiance'])
            hum = np.random.uniform(*feature_ranges[season]['humidity'])
            wind = np.random.uniform(*feature_ranges[season]['wind_speed'])
            temp = np.random.uniform(*feature_ranges[season]['ambient_temperature'])
            tilt = np.random.uniform(*feature_ranges[season]['tilt_angle'])

            kwh = calculate_kwh(season, irr, hum, wind, temp, tilt)

            data.append({
                'irradiance': round(irr, 2),
                'humidity': round(hum, 2),
                'wind_speed': round(wind, 2),
                'ambient_temperature': round(temp, 2),
                'tilt_angle': round(tilt, 2),
                'kwh': round(kwh, 2),
                'season': season,
                'month': month
            })
    return pd.DataFrame(data)

# Generate summer data matching days in each month
df_summer = generate_season_data_by_month('summer', feature_ranges, summer_months_days)

# Generate winter data matching days in each month
df_winter = generate_season_data_by_month('winter', feature_ranges, winter_months_days)

# Generate monsoon data matching days in each month
df_monsoon = generate_season_data_by_month('monsoon', feature_ranges, monsoon_months_days)

print(df_summer.head())
print(f'Total summer data points generated: {len(df_summer)}')  # Should be 31+30+31+30=122

print(df_winter.head())
print(f'Total winter data points generated: {len(df_winter)}')  # Should be 30+31+31+28=120

print(df_monsoon.head())
print(f'Total monsoon data points generated: {len(df_monsoon)}')  # Should be 31+31+30+31=123

# Concatenate summer, winter, and monsoon dataframes
df_all_seasons = pd.concat([df_summer, df_winter, df_monsoon], ignore_index=True)

# Show first few rows to verify
# .head() -> first 5
# .tail() -> last 5
# .info() -> info about data frame
# .describe() -> features like max, min, 50%, 25%, 75% and other stats for each column
#print(df_all_seasons.head())

# Print dataset info
df_all_seasons.info()

# Save it to csv
df_all_seasons.to_csv('solar_performance_all_seasons.csv', index=False)

# Save the data
df_all_seasons.to_pickle('df_all_seasons.pkl')