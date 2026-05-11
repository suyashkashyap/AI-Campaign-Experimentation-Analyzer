# AI Campaign Experimentation Analyzer — Python Notebook Code

```python
# ============================================================
# AI Campaign Experimentation Analyzer
# Author: Suyash Kashyap
# Purpose:
# Automate A/B test analysis for marketing campaign creatives
# ============================================================

# Core libraries for data analysis
import pandas as pd
import numpy as np

# Statistical testing library
from statsmodels.stats.proportion import proportions_ztest

# ------------------------------------------------------------
# Generate synthetic marketing campaign test data
# ------------------------------------------------------------

np.random.seed(42)

creatives = ["Creative_A","Creative_B","Creative_C"]
segments = ["Segment_1","Segment_2","Segment_3"]
status = ["Test","Control"]

rows = []

for i in range(300):
    creative = np.random.choice(creatives)
    segment = np.random.choice(segments)
    test_control = np.random.choice(status)

    snd = np.random.randint(5000, 15000)
    opn = np.random.randint(int(snd*0.15), int(snd*0.45))
    clk = np.random.randint(int(opn*0.08), int(opn*0.30))
    app = np.random.randint(int(clk*0.05), int(clk*0.20))
    response = np.random.randint(int(app*0.50), app+1)

    rows.append([
        creative,
        segment,
        test_control,
        snd,
        opn,
        clk,
        app,
        response
    ])

# Create DataFrame

df = pd.DataFrame(rows, columns=[
    "Creative_name",
    "Segment",
    "Test_control_tagging",
    "snd",
    "opn",
    "clk",
    "app",
    "response"
])

print(df.head())

# ------------------------------------------------------------
# Calculate campaign funnel metrics
# ------------------------------------------------------------

df["Open_Rate"] = df["opn"] / df["snd"]

df["Click_Rate"] = df["clk"] / df["snd"]

df["App_rate"] = df["app"] / df["clk"]

df["Response_rate"] = df["response"] / df["app"]

print(df[[
    "Creative_name",
    "Open_Rate",
    "Click_Rate",
    "App_rate",
    "Response_rate"
]].head())

# ------------------------------------------------------------
# Detect winning creatives using statistical significance
# ------------------------------------------------------------

creative_summary = df.groupby(
    ["Creative_name","Test_control_tagging"]
).agg({
    "snd":"sum",
    "app":"sum"
}).reset_index()

print(creative_summary)

results = []

for creative in creatives:

    creative_data = creative_summary[
        creative_summary["Creative_name"] == creative
    ]

    if len(creative_data) == 2:

        test_row = creative_data[
            creative_data["Test_control_tagging"] == "Test"
        ]

        control_row = creative_data[
            creative_data["Test_control_tagging"] == "Control"
        ]

        if not test_row.empty and not control_row.empty:

            successes = np.array([
                test_row["app"].values[0],
                control_row["app"].values[0]
            ])

            observations = np.array([
                test_row["snd"].values[0],
                control_row["snd"].values[0]
            ])

            stat, pval = proportions_ztest(successes, observations)

            test_rate = successes[0] / observations[0]
            control_rate = successes[1] / observations[1]

            winner = "Test" if test_rate > control_rate else "Control"

            significance = "Significant" if pval < 0.05 else "Not Significant"

            results.append([
                creative,
                round(test_rate,4),
                round(control_rate,4),
                round(pval,5),
                winner,
                significance
            ])

results_df = pd.DataFrame(results, columns=[
    "Creative",
    "Test_Conversion_Rate",
    "Control_Conversion_Rate",
    "P_Value",
    "Winner",
    "Significance"
])

print(results_df)

# ------------------------------------------------------------
# Detect which segment shows strongest lift
# ------------------------------------------------------------

segment_summary = df.groupby(
    ["Segment","Test_control_tagging"]
).agg({
    "snd":"sum",
    "app":"sum"
}).reset_index()

segment_results = []

for segment in segments:

    segment_data = segment_summary[
        segment_summary["Segment"] == segment
    ]

    if len(segment_data) == 2:

        test_row = segment_data[
            segment_data["Test_control_tagging"] == "Test"
        ]

        control_row = segment_data[
            segment_data["Test_control_tagging"] == "Control"
        ]

        if not test_row.empty and not control_row.empty:

            test_rate = (
                test_row["app"].values[0] /
                test_row["snd"].values[0]
            )

            control_rate = (
                control_row["app"].values[0] /
                control_row["snd"].values[0]
            )

            lift = ((test_rate - control_rate) / control_rate) * 100

            segment_results.append([
                segment,
                round(test_rate,4),
                round(control_rate,4),
                round(lift,2)
            ])

segment_results_df = pd.DataFrame(segment_results, columns=[
    "Segment",
    "Test_Rate",
    "Control_Rate",
    "Lift_Percentage"
])

print(segment_results_df)

# ------------------------------------------------------------
# Generate automated experiment insights
# This simulates an "AI analyst"
# ------------------------------------------------------------

insights = []

winner_creatives = results_df[
    results_df["Winner"] == "Test"
]

for _, row in winner_creatives.iterrows():

    insight = f"{row['Creative']} test variant outperformed control with conversion uplift and p-value {row['P_Value']}."

    insights.append(insight)

best_segment = segment_results_df.sort_values(
    by="Lift_Percentage",
    ascending=False
).iloc[0]

segment_insight = (
    f"{best_segment['Segment']} showed highest performance lift of "
    f"{best_segment['Lift_Percentage']}% during experimentation."
)

insights.append(segment_insight)

print("\n================ AI GENERATED INSIGHTS ================\n")

for i in insights:
    print("-", i)

# ------------------------------------------------------------
# Export results
# ------------------------------------------------------------

results_df.to_csv("creative_experiment_results.csv", index=False)
segment_results_df.to_csv("segment_level_results.csv", index=False)

print("\nFiles exported successfully.")

# ------------------------------------------------------------
# Advanced AI-like Campaign Analytics Copilot (No API Needed)
# ------------------------------------------------------------

print("\n================ CAMPAIGN ANALYTICS COPILOT ================\n")

overall_ctr = round(df["Click_Rate"].mean()*100,2)
overall_open = round(df["Open_Rate"].mean()*100,2)
overall_response = round(df["Response_rate"].mean()*100,2)

print(f"Overall Open Rate: {overall_open}%")
print(f"Overall CTR: {overall_ctr}%")
print(f"Overall Response Rate: {overall_response}%")

best_creative = results_df.sort_values(
    by="Test_Conversion_Rate",
    ascending=False
).iloc[0]

print(
    f"\nBest Performing Creative: {best_creative['Creative']}"
)

print(
    f"Test Conversion Rate: {round(best_creative['Test_Conversion_Rate']*100,2)}%"
)

poor_segments = segment_results_df[
    segment_results_df["Lift_Percentage"] < 0
]

if len(poor_segments) > 0:

    print("\nSegments Needing Optimization:")

    for _, row in poor_segments.iterrows():
        print(
            f"- {row['Segment']} showing negative lift of {row['Lift_Percentage']}%"
        )

else:
    print("\nAll segments showing positive lift.")

print("\n================ END OF ANALYSIS ================")
```

