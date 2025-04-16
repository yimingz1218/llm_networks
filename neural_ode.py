import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from sklearn.metrics import mean_squared_error

# Check for MPS availability (Apple Silicon GPU)
def get_device():
    """
    Get the appropriate device for computation.
    Prioritizes MPS (Apple Silicon GPU) if available,
    falls back to CUDA, then CPU.
    
    Returns:
        torch.device: The selected computation device
    """
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class MeanFieldDynamics(nn.Module):
    """
    Mean field dynamics for LLM network modeling.
    
    This class focuses solely on learning the mean field dynamics:
    dρ/dt = F(q, u)ρ
    
    where:
    - ρ is the latent state distribution (Truth, Hallucination, Don't Know)
    - q is the degree distribution (derived from adjacency matrix)
    - u is a control variable (optional)
    - F is a matrix-valued function we're learning
    """
    def __init__(self, latent_dim, hidden_dim=64, degree_dim=1, control_dim=0):
        super().__init__()
        
        # Dimensions
        self.latent_dim = latent_dim  # Dimension of latent state (typically 3 for T, H, D)
        self.degree_dim = degree_dim  # Dimension of degree distribution
        self.control_dim = control_dim  # Dimension of control variable
        
        # Input dimension includes only degree and control, NOT the state itself
        # This ensures F is only a function of q and u, not ρ
        input_dim = 0
        if degree_dim is not None:
            input_dim += degree_dim
        if control_dim > 0:
            input_dim += control_dim
            
        # If no inputs provided, create a static F matrix
        if input_dim == 0:
            self.F = nn.Parameter(torch.zeros(latent_dim, latent_dim, dtype=torch.float32))
            self.static_mode = True
        else:
            # Network to learn the F matrix as a function of q and u
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim * latent_dim, dtype=torch.float32)  # Output is flattened F matrix
            )
            self.static_mode = False
        
    def compute_F_matrix(self, q=None, u=None):
        """
        Compute the F matrix based on degree distribution and control.
        
        Args:
            q: Degree distribution [batch_size, degree_dim]
            u: Control variable [batch_size, control_dim]
            
        Returns:
            F: The transition matrix [batch_size, latent_dim, latent_dim]
        """
        if self.static_mode:
            # If static mode, return the learned F matrix directly
            batch_size = 1 if q is None else q.shape[0]
            return self.F.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create input by concatenating degree distribution and control
        inputs = []
        if q is not None:
            inputs.append(q)
        if u is not None and self.control_dim > 0:
            inputs.append(u)
            
        input_tensor = torch.cat(inputs, dim=1)
        
        # Get the F matrix by reshaping the network output
        F_flat = self.net(input_tensor)
        batch_size = input_tensor.shape[0]
        F = F_flat.view(batch_size, self.latent_dim, self.latent_dim)
        
        return F
        
    def forward(self, t, y, q=None, u=None):
        """
        Compute dρ/dt = F(q, u)ρ
        
        Args:
            t: Time (scalar)
            y: Current state ρ [batch_size, latent_dim]
            q: Degree distribution (optional) [batch_size, degree_dim]
            u: Control variable (optional) [batch_size, control_dim]
            
        Returns:
            dρ/dt: Rate of change of latent state [batch_size, latent_dim]
        """
        batch_size = y.shape[0]
        
        # Compute F matrix based on q and u
        F = self.compute_F_matrix(q, u)
        
        # Compute dρ/dt = F(q, u)ρ using batch matrix multiplication
        dy_dt = torch.bmm(F, y.unsqueeze(2)).squeeze(2)
        
        return dy_dt


class MeanFieldModel(nn.Module):
    """
    Mean field model for LLM network dynamics.
    
    This model uses Neural ODEs to learn and predict the evolution of latent states
    according to the mean field approximation: dρ/dt = F(q, u)ρ
    """
    def __init__(self, latent_dim, hidden_dim=64, degree_dim=1, control_dim=0):
        super().__init__()
        
        self.ode_func = MeanFieldDynamics(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            degree_dim=degree_dim,
            control_dim=control_dim
        )
        
    def forward(self, initial_state, time_points, q=None, u=None):
        """
        Predict latent state evolution using the mean field dynamics.
        
        Args:
            initial_state: Initial latent state ρ(0) [batch_size, latent_dim]
            time_points: Time points to evaluate [num_time_points]
            q: Degree distribution (optional) [batch_size, degree_dim]
            u: Control variable (optional) [batch_size, control_dim]
            
        Returns:
            Predicted latent states at specified time points [num_time_points, batch_size, latent_dim]
        """
        # If q and u are provided, we need to incorporate them into the ODE function
        if q is not None or u is not None:
            # Create a closure over q and u
            def ode_func_with_parameters(t, y):
                return self.ode_func(t, y, q, u)
            
            # Integrate with the parameterized ODE function
            return odeint(ode_func_with_parameters, initial_state, time_points)
        else:
            # Integrate without additional parameters
            return odeint(self.ode_func, initial_state, time_points)
            
    def get_F_matrix(self, q=None, u=None):
        """
        Get the F matrix for the mean field dynamics F(q, u).
        
        Args:
            q: Degree distribution (optional) [batch_size, degree_dim]
            u: Control variable (optional) [batch_size, control_dim]
            
        Returns:
            F: The learned F matrix [batch_size, latent_dim, latent_dim]
        """
        return self.ode_func.compute_F_matrix(q, u)


def compute_degree_distribution(adjacency_matrix):
    """
    Compute the degree distribution from the adjacency matrix.
    
    Args:
        adjacency_matrix: Binary adjacency matrix [N, N]
        
    Returns:
        Degree counts for each node [N]
    """
    # Compute out-degree (sum along rows)
    out_degrees = np.sum(adjacency_matrix, axis=1)
    return out_degrees


def preprocess_data(adjacency_matrix, latent_states):
    """
    Preprocess the data for training the neural ODE.
    
    Args:
        adjacency_matrix: Binary adjacency matrix [N, N]
        latent_states: Latent state distribution over time [num_timesteps, N, 3]
        
    Returns:
        processed_data: Dictionary containing processed tensors
    """
    # Ensure input data is float32
    if isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = adjacency_matrix.astype(np.float32)
    if isinstance(latent_states, np.ndarray):
        latent_states = latent_states.astype(np.float32)
    
    num_timesteps, num_agents, num_states = latent_states.shape
    
    # Create time points
    time_points = np.linspace(0, 1, num_timesteps, dtype=np.float32)
    
    # Compute degree distribution
    degree_dist = compute_degree_distribution(adjacency_matrix)
    
    # Convert to PyTorch tensors with explicit float32 dtype
    time_points_tensor = torch.tensor(time_points, dtype=torch.float32)
    latent_states_tensor = torch.tensor(latent_states, dtype=torch.float32)
    degree_dist_tensor = torch.tensor(degree_dist, dtype=torch.float32)
    
    # Reshape latent states for training
    # [num_timesteps, N, 3] -> [N, num_timesteps, 3]
    latent_states_reshaped = latent_states_tensor.permute(1, 0, 2)
    
    return {
        'time_points': time_points_tensor,
        'latent_states': latent_states_tensor,
        'latent_states_reshaped': latent_states_reshaped,
        'degree_dist': degree_dist_tensor,
        'num_agents': num_agents,
        'num_states': num_states,
        'num_timesteps': num_timesteps
    }


def train_model(model, data, num_epochs=200, learning_rate=0.01, batch_size=None):
    """
    Train the Neural ODE model on the latent state data.
    
    Args:
        model: MeanFieldModel instance
        data: Preprocessed data dictionary
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training (None = use all agents)
        
    Returns:
        trained_model: Trained model
        loss_history: Training loss history
    """
    device = get_device()
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Unpack data and move to device with explicit float32 type
    time_points = data['time_points'].to(device, dtype=torch.float32)
    latent_states = data['latent_states_reshaped'].to(device, dtype=torch.float32)
    degree_dist = data['degree_dist'].to(device, dtype=torch.float32)
    num_agents = data['num_agents']
    
    # Use all agents if batch_size is None
    if batch_size is None:
        batch_size = num_agents
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Training loop
    loss_history = []
    
    for epoch in range(num_epochs):
        # Randomly select agents for batch
        if batch_size < num_agents:
            batch_indices = np.random.choice(num_agents, batch_size, replace=False)
        else:
            batch_indices = np.arange(num_agents)
            
        # Get batch data
        latent_states_batch = latent_states[batch_indices]
        degree_dist_batch = degree_dist[batch_indices].unsqueeze(1) if degree_dist is not None else None
        
        # Initial states for the batch
        initial_states_batch = latent_states_batch[:, 0, :]
        
        # Forward pass: predict latent states
        predicted_states = model(
            initial_states_batch, 
            time_points, 
            q=degree_dist_batch
        )
        
        # Reshape predictions to match target:
        # [num_timesteps, batch_size, 3] -> [batch_size, num_timesteps, 3]
        predicted_states = predicted_states.permute(1, 0, 2)
        
        # Compute loss
        loss = loss_fn(predicted_states, latent_states_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')
    
    return model, loss_history


def evaluate_model(model, data):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained MeanFieldModel
        data: Preprocessed data dictionary
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Model predictions
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Unpack data
    time_points = data['time_points'].to(device, dtype=torch.float32)
    latent_states = data['latent_states_reshaped'].to(device, dtype=torch.float32)
    degree_dist = data['degree_dist'].to(device, dtype=torch.float32)
    num_agents = data['num_agents']
    
    # Get initial states
    initial_states = latent_states[:, 0, :]
    
    # Make predictions
    with torch.no_grad():
        predicted_states = model(
            initial_states, 
            time_points, 
            q=degree_dist.unsqueeze(1) if degree_dist is not None else None
        )
        
    # Reshape predictions to match target
    predicted_states = predicted_states.permute(1, 0, 2)
    
    # Compute metrics
    mse = nn.MSELoss()(predicted_states, latent_states).item()
    
    # Convert to numpy for additional metrics
    pred_np = predicted_states.cpu().numpy()
    true_np = latent_states.cpu().numpy()
    
    # Calculate RMSE for each latent state
    rmse_per_state = np.sqrt(np.mean((pred_np - true_np)**2, axis=(0, 1)))
    
    metrics = {
        'mse': mse,
        'rmse_per_state': rmse_per_state
    }
    
    return metrics, predicted_states


def visualize_mean_field_dynamics(data, predictions, parameters, num_agents_to_plot=3):
    """
    Visualize the mean field dynamics and prediction results.
    
    Args:
        data: Preprocessed data dictionary
        predictions: Model predictions [num_agents, num_timesteps, num_states]
        parameters: Extracted mean field parameters
        num_agents_to_plot: Number of agents to include in visualization
    """
    # Convert predictions to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Get true latent states and time points
    true_states = data['latent_states_reshaped']
    if torch.is_tensor(true_states):
        true_states = true_states.cpu().numpy()
    
    time_points = data['time_points']
    if torch.is_tensor(time_points):
        time_points = time_points.cpu().numpy()
    
    # Create more informative labels for the states
    state_labels = ['Truth (T)', 'Hallucination (H)', 'Don\'t Know (D)']
    
    # Select a subset of agents to plot
    agent_indices = np.random.choice(
        data['num_agents'], 
        min(num_agents_to_plot, data['num_agents']), 
        replace=False
    )
    
    # Plot settings
    fig, axes = plt.subplots(len(agent_indices), 3, figsize=(15, 4 * len(agent_indices)))
    
    for i, agent_idx in enumerate(agent_indices):
        for j in range(3):  # For each latent state
            ax = axes[i, j] if len(agent_indices) > 1 else axes[j]
            
            # Plot true values
            ax.plot(time_points, true_states[agent_idx, :, j], 'b-', label='True')
            
            # Plot predicted values
            ax.plot(time_points, predictions[agent_idx, :, j], 'r--', label='Predicted')
            
            # Add labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Probability')
            ax.set_title(f'Agent {agent_idx+1}, {state_labels[j]}')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the F matrices and their eigenvalues
    f_matrices = parameters['f_matrices']
    degree_values = parameters['degree_values']
    eigenvalues = parameters['eigenvalues']
    dominant_eigenvalues = parameters['dominant_eigenvalues']
    
    # Plot F matrices for selected degree values
    num_matrices_to_plot = min(5, len(f_matrices))
    indices = np.linspace(0, len(f_matrices)-1, num_matrices_to_plot, dtype=int)
    
    fig, axes = plt.subplots(2, num_matrices_to_plot, figsize=(15, 6))
    
    # First row: F matrices
    for i, idx in enumerate(indices):
        ax = axes[0, i]
        im = ax.imshow(f_matrices[idx], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'F matrix (degree={degree_values[idx]:.1f})')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        ax.set_xticks(range(f_matrices[idx].shape[1]))
        ax.set_yticks(range(f_matrices[idx].shape[0]))
        ax.set_xticklabels(['T', 'H', 'D'])
        ax.set_yticklabels(['T', 'H', 'D'])
        
    # Add colorbar
    fig.colorbar(im, ax=axes[0, :])
    
    # Second row: Eigenvalues in complex plane
    for i, idx in enumerate(indices):
        ax = axes[1, i]
        eig_vals = eigenvalues[idx]
        ax.scatter(np.real(eig_vals), np.imag(eig_vals), c='b', marker='o')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Eigenvalues (degree={degree_values[idx]:.1f})')
        ax.grid(True, alpha=0.3)
        
        # Add circle to show unit circle
        unit_circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--', alpha=0.3)
        ax.add_artist(unit_circle)
    
    plt.tight_layout()
    plt.show()
    
    # Plot how F matrix elements change with degree
    latent_dim = f_matrices.shape[1]
    
    fig, axes = plt.subplots(latent_dim, latent_dim, figsize=(12, 10), sharex=True)
    
    for i in range(latent_dim):
        for j in range(latent_dim):
            ax = axes[i, j]
            
            # Extract the (i,j) element across all F matrices
            f_ij_values = f_matrices[:, i, j]
            
            # Plot F(i,j) vs degree
            ax.plot(degree_values, f_ij_values, 'o-')
            
            # If we have trend information, plot the trend line
            if parameters.get('f_trends') is not None and (i, j) in parameters['f_trends']:
                trend = parameters['f_trends'][(i, j)]
                x_fine = np.linspace(min(degree_values), max(degree_values), 100)
                
                if trend['fit_type'] == 'linear':
                    y_fit = np.polyval(trend['coefficients'], x_fine)
                    ax.plot(x_fine, y_fit, 'r--', linewidth=1, alpha=0.7)
                    
                elif trend['fit_type'] == 'quadratic':
                    y_fit = np.polyval(trend['coefficients'], x_fine)
                    ax.plot(x_fine, y_fit, 'g--', linewidth=1, alpha=0.7)
                    
                elif trend['fit_type'] == 'exponential':
                    y_fit = np.exp(np.polyval(trend['coefficients'], x_fine))
                    ax.plot(x_fine, y_fit, 'b--', linewidth=1, alpha=0.7)
            
            # Label the plot based on state transitions
            states = ['T', 'H', 'D']
            ax.set_title(f'F_{states[i]}{states[j]}')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits to be symmetric if values cross zero
            if min(f_ij_values) < 0 and max(f_ij_values) > 0:
                y_max = max(abs(min(f_ij_values)), abs(max(f_ij_values)))
                ax.set_ylim(-y_max*1.1, y_max*1.1)
    
    # Add common x-axis label
    fig.text(0.5, 0.01, 'Degree', ha='center', va='center', fontsize=12)
    
    # Add overall title
    fig.suptitle('F Matrix Elements vs Degree', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    # Plot dominant eigenvalue vs degree
    plt.figure(figsize=(8, 6))
    plt.plot(degree_values, dominant_eigenvalues, 'o-', linewidth=2)
    plt.xlabel('Degree')
    plt.ylabel('Dominant Eigenvalue (Real Part)')
    plt.title('Stability Analysis: Dominant Eigenvalue vs Degree')
    plt.grid(True)
    
    # Add horizontal line at y=0 to show stability threshold
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Stability Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def extract_mean_field_parameters(model, data):
    """
    Extract and analyze the learned mean field dynamics parameters.
    
    Args:
        model: Trained MeanFieldModel
        data: Preprocessed data dictionary
        
    Returns:
        parameters: Dictionary containing extracted F matrices and analysis
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Number of latent states (typically 3: Truth, Hallucination, Don't Know)
    latent_dim = data['num_states']
    
    # Create a range of degree values to analyze F(q)
    if 'degree_dist' in data:
        # Get unique degree values from the data
        degree_values = data['degree_dist'].unique().to(device)
        degree_values, _ = torch.sort(degree_values)
        
        # If too many, sample a reasonable number
        if len(degree_values) > 10:
            indices = torch.linspace(0, len(degree_values)-1, 10).long()
            degree_values = degree_values[indices]
            
        # Create tensor of degree values for analysis
        degree_samples = degree_values.unsqueeze(1)  # [num_samples, 1]
    else:
        # If no degree information, use a default range
        degree_samples = torch.linspace(1, 10, 10, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Extract F matrices for different degree values
    with torch.no_grad():
        # Get F matrices for each degree value
        f_matrices = model.get_F_matrix(q=degree_samples)  # [num_samples, latent_dim, latent_dim]
    
    # Convert to numpy for analysis
    f_matrices_np = f_matrices.cpu().numpy()
    degree_values_np = degree_samples.cpu().numpy().flatten()
    
    # Analyze eigenvalues of F matrices
    eigenvalues = []
    dominant_eigenvalues = []
    
    for i, f_matrix in enumerate(f_matrices_np):
        # Compute eigenvalues
        eig_vals, eig_vecs = np.linalg.eig(f_matrix)
        
        # Sort by magnitude (descending)
        idx = np.argsort(-np.abs(eig_vals))
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        eigenvalues.append(eig_vals)
        
        # Store the dominant eigenvalue (largest real part)
        dominant_eigenvalues.append(np.real(eig_vals[0]))
    
    # Analyze how F changes with degree
    f_sensitivity = np.zeros((latent_dim, latent_dim))
    for i in range(latent_dim):
        for j in range(latent_dim):
            # Get the (i,j) element of F across all degree values
            f_ij_values = f_matrices_np[:, i, j]
            
            # Compute correlation with degree
            if len(degree_values_np) >= 2:  # Need at least 2 points for correlation
                correlation = np.corrcoef(degree_values_np, f_ij_values)[0, 1]
                f_sensitivity[i, j] = correlation
    
    # Functional form of F with respect to degree
    if len(degree_values_np) >= 3:  # Need at least 3 points to estimate trend
        f_trends = {}
        for i in range(latent_dim):
            for j in range(latent_dim):
                f_ij_values = f_matrices_np[:, i, j]
                
                # Check if relationship is approximately linear, quadratic, or exponential
                # Linear: y = ax + b
                # Quadratic: y = ax^2 + bx + c
                # Exponential: y = a*e^(bx)
                
                # Linear fit
                linear_coef = np.polyfit(degree_values_np, f_ij_values, 1)
                linear_fit = np.polyval(linear_coef, degree_values_np)
                linear_error = np.mean((f_ij_values - linear_fit)**2)
                
                # Quadratic fit
                quad_coef = np.polyfit(degree_values_np, f_ij_values, 2)
                quad_fit = np.polyval(quad_coef, degree_values_np)
                quad_error = np.mean((f_ij_values - quad_fit)**2)
                
                # Exponential fit (if all values are positive)
                if np.all(f_ij_values > 0):
                    try:
                        log_y = np.log(f_ij_values)
                        exp_coef = np.polyfit(degree_values_np, log_y, 1)
                        exp_fit = np.exp(np.polyval(exp_coef, degree_values_np))
                        exp_error = np.mean((f_ij_values - exp_fit)**2)
                    except:
                        exp_error = float('inf')
                else:
                    exp_error = float('inf')
                
                # Determine best fit
                best_error = min(linear_error, quad_error, exp_error)
                if best_error == linear_error:
                    fit_type = "linear"
                    coef = linear_coef
                elif best_error == quad_error:
                    fit_type = "quadratic"
                    coef = quad_coef
                else:
                    fit_type = "exponential"
                    coef = exp_coef
                
                f_trends[(i, j)] = {
                    "fit_type": fit_type,
                    "coefficients": coef,
                    "error": best_error
                }
    else:
        f_trends = None
    
    # Compile results
    parameters = {
        'degree_values': degree_values_np,
        'f_matrices': f_matrices_np,
        'eigenvalues': eigenvalues,
        'dominant_eigenvalues': dominant_eigenvalues,
        'f_sensitivity': f_sensitivity,
        'f_trends': f_trends
    }
    
    return parameters


def main(adjacency_matrix, latent_states):
    """
    Main function to run the parameter estimation.
    
    Args:
        adjacency_matrix: Binary adjacency matrix [N, N]
        latent_states: Latent state distribution over time [num_timesteps, N, 3]
        
    Returns:
        model: Trained model
        results: Dictionary containing evaluation results and parameters
    """
    # Print device information
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "mps":
        print("MPS (Metal Performance Shaders) is being used for acceleration")
    elif device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Ensure input data is float32
    if isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = adjacency_matrix.astype(np.float32)
    if isinstance(latent_states, np.ndarray):
        latent_states = latent_states.astype(np.float32)
    # Preprocess the data
    data = preprocess_data(adjacency_matrix, latent_states)
    
    # Create the model
    model = MeanFieldModel(
        latent_dim=data['num_states'],
        hidden_dim=64,
        degree_dim=1  # Using scalar degree information
    )
    
    # Train the model
    trained_model, loss_history = train_model(
        model=model,
        data=data,
        num_epochs=300,
        learning_rate=0.001
    )
    
    # Evaluate the model
    metrics, predictions = evaluate_model(trained_model, data)
    parameters = extract_mean_field_parameters(trained_model, data)

    # Visualize results
    visualize_mean_field_dynamics(data, predictions, parameters)
    
    # Extract parameters
    
    # Compile results
    results = {
        'metrics': metrics,
        'loss_history': loss_history,
        'predictions': predictions,
        'parameters': parameters
    }
    
    return trained_model, results


# Example usage:
if __name__ == "__main__":
    # Load or generate your data here
    # For demonstration purposes, let's generate synthetic data
    
    # Example parameters
    N = 100  # Number of agents
    T = 1000  # Number of time steps
    adj = np.load("adjacencyandnetworkdynamics/adj.npy")
    opinions = np.load("adjacencyandnetworkdynamics/array_100x1000.npy") ### 100 x 1000 array of latent states (1,-1,0)

        ### create a 1000 x 3 array of latent states (1,-1,0) containing the proportions of each latent state in the network per degree
    latent_states = np.zeros((1000,100,3))
    for i in range(1000):
        for k in range(100):
            for j in [-1,0,1]:
            
                ### agents with degree k
                agents_with_degree_k = np.where(adj.sum(axis=1)==k)[0]
                latent_states[i,k,j] = np.mean(opinions[agents_with_degree_k,i]==j)

    ### fill in the missing values
    latent_states = np.nan_to_num(latent_states,0)


    # Run the parameter estimation
    model, results = main(adj, latent_states)
    
    print(f"Final MSE: {results['metrics']['mse']:.6f}")
    print(f"RMSE per state: {results['metrics']['rmse_per_state']}")