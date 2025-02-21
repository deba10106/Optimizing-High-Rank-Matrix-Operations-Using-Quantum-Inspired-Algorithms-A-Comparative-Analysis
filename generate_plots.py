import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.5)

# Data from our experiments
ranks = [10, 20, 50, 100]

# Random Matrix Results
random_basic_error = [2.132e+02, 2.282e+02, 2.272e+02, 2.553e+02]
random_quantum_error = [1.094e-15, 2.413e-15, 2.759e-15, 4.252e-11]
random_basic_time = [0.014, 0.023, 0.063, 0.079]
random_quantum_time = [0.018, 0.115, 0.553, 1.021]

# Portfolio Matrix Results
portfolio_basic_error = [1.109e+05, 1.235e+05, 1.365e+05, 1.732e+05]
portfolio_quantum_error = [3.246e-02, 2.400e-02, 1.889e-02, 1.063e-01]
portfolio_basic_time = [0.011, 0.016, 0.060, 0.025]
portfolio_quantum_time = [0.019, 0.088, 0.581, 0.968]

# Recommendation Matrix Results
recom_basic_error = [5.379e+02, 7.844e+02, 2.135e+03, 5.293e+03]
recom_quantum_error = [9.668e-01, 9.636e-01, 9.327e-01, 8.798e-01]
recom_basic_time = [0.009, 0.011, 0.095, 0.024]
recom_quantum_time = [0.016, 0.020, 0.651, 1.004]

def plot_comparison(ranks, basic_data, quantum_data, title, ylabel, yscale='log'):
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, basic_data, 'o-', label='Basic FKV', color='#1f77b4', linewidth=2, markersize=8)
    plt.plot(ranks, quantum_data, 's-', label='Quantum-Inspired', color='#d62728', linewidth=2, markersize=8)
    plt.xlabel('Matrix Rank')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.yscale(yscale)
    plt.legend()
    return plt

# Plot 1: Error Comparison Across Scenarios
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(ranks, random_basic_error, 'o-', label='Basic FKV', color='#1f77b4')
plt.plot(ranks, random_quantum_error, 's-', label='Quantum-Inspired', color='#d62728')
plt.yscale('log')
plt.title('Random Matrices')
plt.xlabel('Rank')
plt.ylabel('Error (log scale)')
plt.legend()

plt.subplot(132)
plt.plot(ranks, portfolio_basic_error, 'o-', label='Basic FKV', color='#1f77b4')
plt.plot(ranks, portfolio_quantum_error, 's-', label='Quantum-Inspired', color='#d62728')
plt.yscale('log')
plt.title('Portfolio Matrices')
plt.xlabel('Rank')
plt.ylabel('Error (log scale)')

plt.subplot(133)
plt.plot(ranks, recom_basic_error, 'o-', label='Basic FKV', color='#1f77b4')
plt.plot(ranks, recom_quantum_error, 's-', label='Quantum-Inspired', color='#d62728')
plt.yscale('log')
plt.title('Recommendation Matrices')
plt.xlabel('Rank')
plt.ylabel('Error (log scale)')

plt.tight_layout()
plt.savefig('error_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Time vs Error Trade-off
plt.figure(figsize=(15, 5))

# Random Matrices
plt.subplot(131)
plt.scatter(random_basic_time, random_basic_error, s=100, c='#1f77b4', label='Basic FKV')
plt.scatter(random_quantum_time, random_quantum_error, s=100, c='#d62728', label='Quantum-Inspired')
for i, rank in enumerate(ranks):
    plt.annotate(f'k={rank}', (random_basic_time[i], random_basic_error[i]), xytext=(5,5), textcoords='offset points')
    plt.annotate(f'k={rank}', (random_quantum_time[i], random_quantum_error[i]), xytext=(5,5), textcoords='offset points')
plt.yscale('log')
plt.title('Random Matrices')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (log scale)')
plt.legend()

# Portfolio Matrices
plt.subplot(132)
plt.scatter(portfolio_basic_time, portfolio_basic_error, s=100, c='#1f77b4', label='Basic FKV')
plt.scatter(portfolio_quantum_time, portfolio_quantum_error, s=100, c='#d62728', label='Quantum-Inspired')
for i, rank in enumerate(ranks):
    plt.annotate(f'k={rank}', (portfolio_basic_time[i], portfolio_basic_error[i]), xytext=(5,5), textcoords='offset points')
    plt.annotate(f'k={rank}', (portfolio_quantum_time[i], portfolio_quantum_error[i]), xytext=(5,5), textcoords='offset points')
plt.yscale('log')
plt.title('Portfolio Matrices')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (log scale)')

# Recommendation Matrices
plt.subplot(133)
plt.scatter(recom_basic_time, recom_basic_error, s=100, c='#1f77b4', label='Basic FKV')
plt.scatter(recom_quantum_time, recom_quantum_error, s=100, c='#d62728', label='Quantum-Inspired')
for i, rank in enumerate(ranks):
    plt.annotate(f'k={rank}', (recom_basic_time[i], recom_basic_error[i]), xytext=(5,5), textcoords='offset points')
    plt.annotate(f'k={rank}', (recom_quantum_time[i], recom_quantum_error[i]), xytext=(5,5), textcoords='offset points')
plt.yscale('log')
plt.title('Recommendation Matrices')
plt.xlabel('Time (seconds)')
plt.ylabel('Error (log scale)')

plt.tight_layout()
plt.savefig('time_error_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Scalability Analysis
plt.figure(figsize=(10, 6))
plt.plot(ranks, random_quantum_time, 's-', label='Random', color='#1f77b4', linewidth=2)
plt.plot(ranks, portfolio_quantum_time, 'o-', label='Portfolio', color='#d62728', linewidth=2)
plt.plot(ranks, recom_quantum_time, '^-', label='Recommendation', color='#2ca02c', linewidth=2)
plt.xlabel('Matrix Rank')
plt.ylabel('Time (seconds)')
plt.title('Scalability Analysis of Quantum-Inspired Algorithm')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
