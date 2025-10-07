import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (make sure filename matches)
df = pd.read_csv("fake-reviews.csv")

# Preview data
print(df.head())

# Count missing values
print(df.isnull().sum())

# Add new features
df["char_length"] = df["text_"].apply(len)
df["word_count"] = df["text_"].str.split().apply(len)

# Visualize data
sns.boxplot(x="char_length", y="label", data=df)
plt.title("Character Length of Reviews by Label")
plt.xlabel("Character length")
plt.ylabel("Review label")
plt.show()
