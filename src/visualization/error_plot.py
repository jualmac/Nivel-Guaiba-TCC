import matplotlib.pyplot as plt

# Forecast horizons
horizons = ["1 hour", "12 hours", "1 day", "7 days", "15 days"]

# RMSE means
rmse_data = {
    "XGBoost":  [9.28, 19.67, 32.33, 31.43, 32.92],
    "LightGBM": [8.15, 20.79, 42.81, 32.87, 37.07],
    "LSTM":     [32.11, 52.40, 47.35, 70.42, 44.96],
    "SARIMA":   [112.34, 90.11, 81.98, 77.49, 66.94],
}

# RMSE standard deviations (optional for error bars)
std_data = {
    "XGBoost":  [8.92, 6.62, 19.69, 24.58, 24.16],
    "LightGBM": [8.48, 9.37, 29.72, 20.75, 24.77],
    "LSTM":     [28.47, 40.75, 22.13, 36.25, 16.22],
    "SARIMA":   [31.53, 30.73, 26.80, 23.062, 16.47],
}

# Create figure
plt.figure(figsize=(10, 6))

# Plot each model
for model in rmse_data:
    plt.plot(
        horizons,
        rmse_data[model],
        marker="o",
        linewidth=2,
        label=model
    )

    # # Error bars
    # plt.errorbar(
    #     horizons,
    #     rmse_data[model],
    #     yerr=std_data[model],
    #     fmt='none',
    #     capsize=4
    # )

# Labels and title
plt.xlabel("Forecasting Horizon")
plt.ylabel("RMSE")
plt.title("RMSE Comparison Between Models and Forecasting Horizons")

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()
plt.savefig("data/rmse_comparison.png", bbox_inches="tight")
