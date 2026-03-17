import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Veri
# ------------------------
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * x + 5 + np.random.randn(100) * 2

n = len(x)

# ------------------------
# Model
# ------------------------
a = 0
b = 0
lr = 0.1
epochs = 10000

# ------------------------
# Plot setup
# ------------------------
plt.ion()  # interactive mode
fig, ax = plt.subplots()

ax.scatter(x, y, label="Veri Seti")
line, = ax.plot(x, a * x + b, label="Model", linewidth=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

# ------------------------
# Training + canlı çizim
# ------------------------
for epoch in range(epochs):
    y_pred = a * x + b

    # Gradientler
    da = (2 / n) * np.sum((y_pred - y) * x)
    db = (2 / n) * np.sum(y_pred - y)

    # Update
    a -= lr * da
    b -= lr * db

    # Doğruyu güncelle
    line.set_ydata(a * x + b)
    ax.set_title(f"Tekrar {epoch+1}")

    plt.pause(0.05)

plt.ioff()
plt.show()
