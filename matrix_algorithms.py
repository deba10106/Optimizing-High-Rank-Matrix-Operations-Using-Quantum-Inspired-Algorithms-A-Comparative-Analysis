import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

class MatrixAlgorithms:
    def __init__(self):
        self.xp = np
        
    def generate_test_matrix(self, m, n, rank, condition_number):
        """Generate a test matrix with specified rank and condition number."""
        U, _ = np.linalg.qr(np.random.randn(m, rank))
        V, _ = np.linalg.qr(np.random.randn(n, rank))
        S = np.diag(np.linspace(1, condition_number, rank))
        return U @ S @ V.T

    def standard_qfkv(self, A, r, c, k=None):
        """Standard Quantum-inspired FKV algorithm (Original Implementation)."""
        m, n = A.shape
        if k is None:
            k = min(r, c)
        
        # Uniform random sampling for rows
        row_indices = np.random.choice(m, r, replace=False)
        R = A[row_indices, :]
        
        # Uniform random sampling for columns
        col_indices = np.random.choice(n, c, replace=False)
        C = R[:, col_indices]
        
        # Standard SVD computation
        U_C, S_C, V_C = np.linalg.svd(C, full_matrices=False)
        
        # Truncate to k components
        k = min(k, min(U_C.shape[1], V_C.shape[0]))
        U_k = U_C[:, :k]
        S_k = S_C[:k]
        
        # Direct matrix reconstruction
        B = A[:, col_indices] @ np.linalg.pinv(C) @ (U_k * S_k)
        T = B.T @ A
        
        # Final SVD
        U_T, S_T, V_T = np.linalg.svd(T, full_matrices=False)
        U_final = B @ U_T
        
        return U_final[:, :k], S_T[:k], V_T[:k, :]

    def enhanced_qfkv(self, A, r, c, k=None):
        """Enhanced Quantum-inspired FKV algorithm (Our Implementation)."""
        m, n = A.shape
        if k is None:
            k = min(r, c)
        
        # Quantum-inspired importance sampling for rows
        row_probs = np.linalg.norm(A, axis=1) ** 2
        row_probs /= np.sum(row_probs)
        row_indices = np.random.choice(m, r, p=row_probs, replace=False)
        R = A[row_indices, :] / np.sqrt(r * row_probs[row_indices, None])
        
        # Quantum-inspired importance sampling for columns
        col_probs = np.linalg.norm(R, axis=0) ** 2
        col_probs /= np.sum(col_probs)
        col_indices = np.random.choice(n, c, p=col_probs, replace=False)
        C = R[:, col_indices] / np.sqrt(c * col_probs[col_indices])
        
        # Randomized SVD for better efficiency
        U_C, S_C, V_C = randomized_svd(C, n_components=min(k, min(C.shape)), n_iter=5)
        
        # Truncate to k components and reconstruct with improved numerical stability
        k = min(k, min(U_C.shape[1], V_C.shape[0]))
        U_k = U_C[:, :k]
        S_k = S_C[:k]
        
        # Nyström-inspired reconstruction with QR orthogonalization
        B = A[:, col_indices] @ np.linalg.pinv(C) @ (U_k * S_k)
        Q, _ = np.linalg.qr(B)  # Orthogonalize for stability
        T = Q.T @ A
        
        # Final SVD with randomized method
        U_T, S_T, V_T = randomized_svd(T, n_components=k, n_iter=5)
        U_final = Q @ U_T
        
        return U_final, S_T, V_T

    def reconstruction_error(self, A, U, S, V):
        """Compute reconstruction error."""
        if len(S.shape) == 1:
            S = np.diag(S)
        A_approx = (U @ S) @ V
        return np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')

def run_experiments():
    # Test parameters
    m, n = 1000, 800
    ranks = [10, 20, 50, 100]
    condition_number = 50
    sample_size = 100  # Number of rows/columns to sample
    
    # Results storage
    results = {
        'standard': {'time': [], 'error': []},
        'enhanced': {'time': [], 'error': []}
    }
    
    algo = MatrixAlgorithms()
    
    # Test different scenarios
    scenarios = [
        ('Random Matrix', lambda: algo.generate_test_matrix(m, n, rank, condition_number)),
        ('Portfolio Matrix', lambda: generate_portfolio_matrix(m, n, rank)),
        ('Recommendation Matrix', lambda: generate_recommendation_matrix(m, n, rank))
    ]
    
    for scenario_name, matrix_generator in scenarios:
        print(f"\nTesting {scenario_name}:")
        
        for rank in ranks:
            print(f"Running experiment for rank {rank}")
            A = matrix_generator()
            
            # Standard FKV
            start_time = time.time()
            U, S, V = algo.standard_qfkv(A, r=sample_size, c=sample_size, k=rank)
            standard_time = time.time() - start_time
            standard_error = float(algo.reconstruction_error(A, U, S, V))
            
            # Enhanced FKV
            start_time = time.time()
            U, S, V = algo.enhanced_qfkv(A, r=sample_size, c=sample_size, k=rank)
            enhanced_time = time.time() - start_time
            enhanced_error = float(algo.reconstruction_error(A, U, S, V))
            
            results['standard']['time'].append(standard_time)
            results['standard']['error'].append(standard_error)
            results['enhanced']['time'].append(enhanced_time)
            results['enhanced']['error'].append(enhanced_error)
            
            print(f"Rank {rank}:")
            print(f"  Standard FKV: {standard_time:.3f}s, Error: {standard_error:.3e}")
            print(f"  Enhanced FKV: {enhanced_time:.3f}s, Error: {enhanced_error:.3e}")
        
        # Plotting for this scenario
        plot_results(ranks, results, scenario_name)
        results = {'standard': {'time': [], 'error': []}, 'enhanced': {'time': [], 'error': []}}

def generate_portfolio_matrix(m, n, rank):
    """Generate a synthetic portfolio matrix with realistic properties."""
    # Create correlated asset returns
    factors = np.random.randn(rank, n)  # Factor matrix
    loadings = np.random.randn(m, rank)  # Factor loadings
    noise = np.random.randn(m, n) * 0.1  # Small noise component
    return loadings @ factors + noise

def generate_recommendation_matrix(m, n, rank):
    """Generate a synthetic recommendation matrix with realistic properties."""
    # Create user and item latent factors
    user_factors = np.random.randn(m, rank)
    item_factors = np.random.randn(n, rank)
    # Add sparsity typical in recommendation systems
    mask = np.random.binomial(1, 0.1, (m, n))  # 90% sparsity
    return (user_factors @ item_factors.T) * mask

def plot_results(ranks, results, scenario_name):
    """Plot comparison results."""
    plt.figure(figsize=(12, 5))
    
    # Runtime comparison
    plt.subplot(1, 2, 1)
    plt.plot(ranks, results['standard']['time'], 'b-o', label='Standard Q-FKV')
    plt.plot(ranks, results['enhanced']['time'], 'r-o', label='Enhanced Q-FKV')
    plt.xlabel('Matrix Rank')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'{scenario_name}\nRuntime Comparison')
    plt.legend()
    plt.grid(True)
    
    # Error comparison
    plt.subplot(1, 2, 2)
    plt.plot(ranks, results['standard']['error'], 'b-o', label='Standard Q-FKV')
    plt.plot(ranks, results['enhanced']['error'], 'r-o', label='Enhanced Q-FKV')
    plt.xlabel('Matrix Rank')
    plt.ylabel('Reconstruction Error')
    plt.title(f'{scenario_name}\nError Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'output_plot_{scenario_name.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    run_experiments()
