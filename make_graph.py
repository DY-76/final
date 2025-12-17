import matplotlib.pyplot as plt
import numpy as np

# Task order and metrics from the latest results
tasks = ["Code Search", "Code Repair", "Code Summary", "Code Generation"]

# Success rate values (percent)
baseline = [13.3, 0.0, 0.0, 0.0]
finetuned = [40.0, 70.0, 83.3, 96.7]

# Extra metrics to annotate where available
extra_metrics = {
    "Code Search": "Accuracy 40.0%",
    "Code Repair": "Pass@1 70.0%",
    "Code Summary": "BLEU-4 0.430",
    "Code Generation": "BLEU-4 0.646",
}

baseline_extra_metrics = {
    "Code Search": "Accuracy 13.3%",
    "Code Repair": "Pass@1 0.0%",
    "Code Summary": "BLEU-4 0.000",
    "Code Generation": "BLEU-4 0.000",
}

x = np.arange(len(tasks))
width = 0.35

plt.figure(figsize=(8, 5))

# Colors
baseline_color = "#87CEFA"   # Light sky blue
finetuned_color = "#00BFFF"  # Deep sky blue

bars1 = plt.bar(x - width / 2, baseline, width, label="Baseline", color=baseline_color)
bars2 = plt.bar(x + width / 2, finetuned, width, label="Fine-tuned", color=finetuned_color)

plt.ylabel("Success Rate (%)")
plt.title("Task Performance Summary")
plt.xticks(x, tasks, rotation=15)
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()

# Add accuracy labels above bars
for bar, task in zip(bars1, tasks):
    height = bar.get_height()
    label = f"{height:.1f}%\n{baseline_extra_metrics.get(task, '')}".rstrip()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

for bar, task in zip(bars2, tasks):
    height = bar.get_height()
    label = f"{height:.1f}%\n{extra_metrics.get(task, '')}".rstrip()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Save figure
plt.savefig("task_performance_summary.png", dpi=300, bbox_inches="tight")

plt.show()

print("Bar chart saved as 'task_performance_summary.png'")
