import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')
class_counts = df['Class'].value_counts()

colors = ['blue', 'red']
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Bar chart ---
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1,
            hue=class_counts.index, palette=colors, legend=False,
            alpha=0.7, edgecolor='#2d3436', linewidth=1)

ax1.set_title('Class Distribution', fontsize=16, fontweight='bold', color='#2d3436')
ax1.set_xlabel('Class', fontsize=14)
ax1.set_ylabel('Count (Log Scale)', fontsize=14)
ax1.set_yscale('log')

for i, v in enumerate(class_counts.values):
    ax1.text(i, v, f'{v}', ha='center', va='bottom', fontsize=12, color='#2d3436')

# --- Pie chart ---
wedges, texts, autotexts = ax2.pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.2f%%',
        colors=colors, explode=(0, 0.1), startangle=45,
        textprops={'color':"#2d3436"},
        wedgeprops={'alpha':0.6, 'edgecolor':'#2d3436', 'linewidth':1})

plt.setp(autotexts, size=12, color="white")
ax2.set_title('Fraud Percentage', fontsize=16, fontweight='bold', color='#2d3436')

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

df = pd.read_csv('creditcard.csv')

sns.set_style("whitegrid")

v_features = df.columns[1:29]

plt.figure(figsize=(30, 15))
gs = gridspec.GridSpec(4, 7)

for i, feature in enumerate(v_features):
    ax = plt.subplot(gs[i])

    # Plot a density plot

    sns.kdeplot(data=df[df['Class'] == 0][feature], label='Normal (0)', color='blue', fill=True, common_norm=False, ax=ax)
    sns.kdeplot(data=df[df['Class'] == 1][feature], label='Fraud (1)', color='red', fill=True, common_norm=False, ax=ax)

    ax.set_xlabel('')
    ax.set_title(f'{feature} Distribution')

    if i == 0:
        ax.legend(loc='best')
    else:
        ax.get_legend().remove() if ax.get_legend() else None

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

data['Hour'] = (data['Time'] % 86400) // 3600

sns.kdeplot(data[data['Class'] == 0]['Hour'], label='Normal (0)', fill=True, color='blue', common_norm=False)
sns.kdeplot(data[data['Class'] == 1]['Hour'], label='Fraud (1)', fill=True, color='red', common_norm=False)

plt.title('Transaction Distribution by Hour', fontsize=15)
plt.xlabel('Relative Hour of Day (0-23)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xticks(range(0, 24))
plt.legend()
plt.show()
