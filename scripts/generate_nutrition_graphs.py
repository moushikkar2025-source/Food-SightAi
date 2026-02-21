"""
FoodSight AI - Nutrition Graph Generator
=========================================

Generates comparative graphs for nutrition data of 35 dishes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from nutrition_data import NUTRITION_DATABASE

# Ensure styles
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Prepare data
data = []
for dish, info in NUTRITION_DATABASE.items():
    nutri = info['nutrition']
    data.append({
        'Dish': dish,
        'Calories': nutri['calories'],
        'Protein (g)': nutri['protein_g'],
        'Carbs (g)': nutri['carbs_g'],
        'Fats (g)': nutri['fats_g'],
        'Fiber (g)': nutri.get('fiber_g', 0)
    })

df = pd.DataFrame(data)

# Sort for better visualization
df_cals = df.sort_values('Calories', ascending=False)
df_protein = df.sort_values('Protein (g)', ascending=False)

# Output directory
OUTPUT_DIR = 'nutrition_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_bar(df_sorted, column, title, filename, color):
    plt.figure(figsize=(12, 10))
    sns.barplot(x=column, y='Dish', data=df_sorted, color=color)
    plt.title(f'{title} Comparison (35 Dishes)', fontsize=16)
    plt.xlabel(title, fontsize=12)
    plt.ylabel('Dish Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    print(f"Generated {filename}")
    plt.close()

# Generate plots
print("Generating nutrition graphs...")
plot_bar(df_cals, 'Calories', 'Calories (kcal)', 'calories_comparison.png', '#e74c3c')
plot_bar(df_protein, 'Protein (g)', 'Protein Content (g)', 'protein_comparison.png', '#3498db')
plot_bar(df.sort_values('Fats (g)', ascending=False), 'Fats (g)', 'Fat Content (g)', 'fat_comparison.png', '#f1c40f')
plot_bar(df.sort_values('Carbs (g)', ascending=False), 'Carbs (g)', 'Carbohydrate Content (g)', 'carbs_comparison.png', '#2ecc71')

print("Done!")
