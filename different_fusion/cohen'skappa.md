cohen_kappa_score only works if labels are numerical. If you have categorical labels like "positive", "neutral", "negative", we need to convert them to integers.

Use Case: Evaluating Annotation Quality in a Human-in-the-Loop AI Pipeline
🧠 Context:
You're working as a Language Engineer or Data Scientist on a team building a natural language understanding (NLU) system—for example, sentiment classification, intent detection, or named entity recognition.

Before training models, you're collecting labeled training data via human annotators using tools like Amazon SageMaker Ground Truth, Labelbox, or an internal annotation portal.

You want to ensure that annotators are consistent and reliable—if two people see the same sentence, they should ideally assign the same label. If not, your training data may be noisy and degrade model performance.

🧪 Objective:
Evaluate inter-annotator agreement using Cohen’s Kappa to detect:

Annotators who are inconsistent with others (low Kappa score).

Labels or samples that are ambiguous.

Whether further training or clarification is needed for your annotation guidelines.

📄 Example Input (annotations.csv):
sample_id	annotator_id	label
001	A	positive
001	B	positive
001	C	neutral
002	A	negative
002	B	negative
002	C	negative
003	A	positive
003	B	neutral
003	C	neutral
✅ What the Script Does:
Loads annotations from a CSV file.

Pivots the data so each row is a sample, and each column is an annotator.

Calculates Cohen’s Kappa for every pair of annotators to measure how similarly they label the same data.

Prints the agreement score between each pair.

📊 Example Output:
vbnet
Copy
Edit
Cohen's Kappa between A and B: 0.33
Cohen's Kappa between A and C: 0.00
Cohen's Kappa between B and C: 0.50
🚩 How This Helps:
If A vs. B is low but B vs. C is high, A might be interpreting labels differently.

Kappa near 1.0 = Strong agreement ✅

Kappa near 0.0 = Random labeling or inconsistent understanding ⚠️

Negative Kappa = Systematic disagreement 🚨

You can use this insight to:

Improve annotation guidelines

Flag low-agreement examples for review

Filter noisy annotations before model training





Part 1: Handle Categorical Labels with Label Encoding
cohen_kappa_score only works if labels are numerical. If you have categorical labels like "positive", "neutral", "negative", we need to convert them to integers.

Here’s how you can update your script:

python
Copy
Edit
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

# Load annotations
df = pd.read_csv("annotations.csv")  # sample_id, annotator_id, label

# Encode categorical labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Pivot to get annotator columns per sample (using encoded labels)
pivot = df.pivot(index="sample_id", columns="annotator_id", values="label_encoded").dropna()

# Pairwise Cohen’s Kappa
annotators = pivot.columns
results = []

for a1, a2 in combinations(annotators, 2):
    kappa = cohen_kappa_score(pivot[a1], pivot[a2])
    results.append((a1, a2, kappa))

# Display results
for a1, a2, score in results:
    print(f"Cohen's Kappa between {a1} and {a2}: {score:.2f}")
🧮 Part 2: Compute Fleiss’ Kappa for >2 Annotators
Fleiss’ Kappa measures agreement when more than 2 annotators label the same items.

We’ll need the data in matrix form, where each row is a sample and each column is the count of how many annotators selected each label.

Here’s a full script for Fleiss’ Kappa:

python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import fleiss_kappa

# Load annotations
df = pd.read_csv("annotations.csv")  # sample_id, annotator_id, label

# Encode labels numerically
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Get label names
labels = list(le.classes_)
num_labels = len(labels)

# Build Fleiss' Kappa matrix: rows = samples, columns = label counts
sample_ids = df['sample_id'].unique()
matrix = np.zeros((len(sample_ids), num_labels), dtype=int)

for i, sid in enumerate(sample_ids):
    sample_df = df[df['sample_id'] == sid]
    for label in sample_df['label_encoded']:
        matrix[i, label] += 1

# Compute Fleiss’ Kappa
fkappa = fleiss_kappa(matrix)

print(f"Fleiss' Kappa: {fkappa:.3f}")