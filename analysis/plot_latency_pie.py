import matplotlib.pyplot as plt

labels = [
    "Attention",
    "MLP",
    "KV Cache",
    "Embedding",
    "Framework"
]

values = [40, 25, 20, 5, 10]  # replace with your actual numbers

plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Latency Breakdown per Token")

plt.savefig("analysis/plots/08_latency_pie.png")