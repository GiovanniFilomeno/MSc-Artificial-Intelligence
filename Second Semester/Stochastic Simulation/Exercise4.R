### -------------------------------------------
### Exercise 4
### Topic 4: Numerical Illustration of the Black-Scholes Model
### -------------------------------------------

# Load necessary packages
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
library(ggplot2)

# Define parameters for the Black-Scholes model
s0 <- 100     # initial stock price
r <- 0.05     # risk-free rate
sigma <- 0.2  # volatility
T <- 1        # time to maturity
N <- 100      # number of time steps
dt <- T / N   # time increment

# Define the time grid
t_grid <- seq(0, T, length.out = N + 1)

##### Part 4.1: Define the exact solution for the Black-Scholes model
exact_solution <- function(s0, t, Wt) {
  s0 * exp((r - 0.5 * sigma^2) * t + sigma * Wt)
}

###### Part 4.2: Implement the Euler scheme
euler_scheme <- function(S, dt, dW) {
  S + r * S * dt + sigma * S * dW
}

##### Part 4.3: Implement the Milstein scheme
milstein_scheme <- function(S, dt, dW) {
  S + r * S * dt + sigma * S * dW + 0.5 * sigma^2 * S * (dW^2 - dt)
}

# Simulate Brownian motion
set.seed(123)  # For reproducibility
W <- cumsum(c(0, sqrt(dt) * rnorm(N))) # Generate increments and accumulate them

# Initialize vectors for storing stock prices
S_exact <- numeric(N + 1)
S_euler <- numeric(N + 1)
S_milstein <- numeric(N + 1)

# Set initial values
S_exact[1] <- s0
S_euler[1] <- s0
S_milstein[1] <- s0

# Simulate the trajectories: Part 4.1, 4.2, 4.3
for (i in 1:N) {
  S_exact[i + 1] <- exact_solution(S_exact[1], t_grid[i + 1], W[i + 1]) # Part 4.1: Implement the exact solution
  S_euler[i + 1] <- euler_scheme(S_euler[i], dt, W[i + 1] - W[i]) # Part 4.2: Implement the euler scheme
  S_milstein[i + 1] <- milstein_scheme(S_milstein[i], dt, W[i + 1] - W[i]) # Part 4.3: Implement the Milstein scheme
}

# Combine results into a data frame for plotting
results <- data.frame(
  Time = rep(t_grid, 3),
  Stock_Price = c(S_exact, S_euler, S_milstein),
  Method = rep(c("Exact", "Euler", "Milstein"), each = N + 1)
)

# Plot the trajectories
p <- ggplot(results, aes(x = Time, y = Stock_Price, color = Method)) +
  geom_line() +
  labs(title = "Black-Scholes Model: Exact, Euler, and Milstein Schemes",
       x = "Time", y = "Stock Price") +
  theme_minimal()

# Save the plot
ggsave("exercise_4_4_trajectories.png", plot = p)

##### Part 4.5: Calculate the empirical mean and L^2 error
M <- 1000  # number of simulations for empirical analysis

# Function to simulate a single trajectory using a specified scheme
simulate_trajectory <- function(scheme, s0, dt, N) {
  S <- numeric(N + 1)
  S[1] <- s0
  W <- cumsum(c(0, sqrt(dt) * rnorm(N)))
  for (i in 1:N) {
    S[i + 1] <- scheme(S[i], dt, W[i + 1] - W[i])
  }
  return(S[N + 1])
}

# Simulate multiple trajectories for Euler scheme
set.seed(123)  # For reproducibility
S_T_exact <- numeric(M)
S_T_euler <- numeric(M)
S_T_milstein <- numeric(M)

for (i in 1:M) {
  S_T_exact[i] <- simulate_trajectory(exact_solution, s0, dt, N)
  S_T_euler[i] <- simulate_trajectory(euler_scheme, s0, dt, N)
  S_T_milstein[i] <- simulate_trajectory(milstein_scheme, s0, dt, N)
}

# Calculate the empirical mean and L^2 error
empirical_mean_euler <- mean((S_T_exact - S_T_euler)^2)
empirical_mean_milstein <- mean((S_T_exact - S_T_milstein)^2)

# Print the results
cat("Empirical mean (Euler):", empirical_mean_euler, "\n")
# Empirical mean (Euler): 865.1899 
cat("Empirical mean (Milstein):", empirical_mean_milstein, "\n")
# Empirical mean (Milstein): 916.3187 

##### Part 4.6: Confidence interval for the empirical mean
alpha <- 0.05  # significance level
z_alpha <- qnorm(1 - alpha / 2) # Z-value for the given significance level

# Calculate confidence intervals for the empirical means
SE_euler <- sqrt(var((S_T_exact - S_T_euler)^2) / M)
CI_euler <- c(empirical_mean_euler - z_alpha * SE_euler, empirical_mean_euler + z_alpha * SE_euler)

SE_milstein <- sqrt(var((S_T_exact - S_T_milstein)^2) / M)
CI_milstein <- c(empirical_mean_milstein - z_alpha * SE_milstein, empirical_mean_milstein + z_alpha * SE_milstein)

# Print the confidence intervals
cat("Confidence interval (Euler): [", CI_euler[1], ", ", CI_euler[2], "]\n")
# Confidence interval (Euler): [ 780.6226 ,  949.7572 ]
cat("Confidence interval (Milstein): [", CI_milstein[1], ", ", CI_milstein[2], "]\n")
# Confidence interval (Milstein): [ 820.5745 ,  1012.063 ]

##### Part 4.7: Plot the evolution of the empirical mean and confidence intervals
empirical_means_euler <- numeric(M)
empirical_means_milstein <- numeric(M)

for (i in 1:M) {
  empirical_means_euler[i] <- mean((S_T_exact[1:i] - S_T_euler[1:i])^2)
  empirical_means_milstein[i] <- mean((S_T_exact[1:i] - S_T_milstein[1:i])^2)
}

# Radius of confidence intervarls
r_M_euler <- z_alpha * SE_euler
r_M_milstein <- z_alpha * SE_milstein

# Calculate the confidence intervals for each simulation count
CI_euler_lower <- empirical_means_euler - r_M_euler
CI_euler_upper <- empirical_means_euler + r_M_euler

CI_milstein_lower <- empirical_means_milstein - r_M_milstein
CI_milstein_upper <- empirical_means_milstein + r_M_milstein

# Create a data frame for plotting
plot_data <- data.frame(
  Simulation = rep(1:M, 2),
  Empirical_Mean = c(empirical_means_euler, empirical_means_milstein),
  Scheme = rep(c("Euler", "Milstein"), each = M)
)

# Create a data frame for the confidence intervals
ci_data <- data.frame(
  Simulation = rep(1:M, 2),
  ymin = c(CI_euler_lower, CI_milstein_lower),
  ymax = c(CI_euler_upper, CI_milstein_upper),
  Scheme = rep(c("Euler", "Milstein"), each = M)
)

# Plot the empirical means and confidence intervals
p2 <- ggplot() +
  geom_line(data = plot_data, aes(x = Simulation, y = Empirical_Mean, color = Scheme)) +
  labs(title = "Evolution of Empirical Mean and Confidence Intervals",
       x = "Simulation", y = "Empirical Mean") +
  geom_ribbon(data = ci_data[ci_data$Scheme == "Euler", ], aes(x = Simulation, ymin = ymin, ymax = ymax), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = ci_data[ci_data$Scheme == "Milstein", ], aes(x = Simulation, ymin = ymin, ymax = ymax), fill = "red", alpha = 0.2) +
  theme_minimal() +
  scale_color_manual(values = c("Euler" = "blue", "Milstein" = "red"))

# Save the plot
ggsave("exercise_4_4_empirical_mean_evolution.png", plot = p2)

##### Part 4.8: Study the L^2 error for the Milstein scheme
cat("L^2 error for Milstein scheme:", empirical_mean_milstein, "\n")
# L^2 error for Milstein scheme: 916.3187 

# Function to compute L^2 error for Milstein scheme as a function of N
compute_L2_error_milstein <- function(s0, r, sigma, T, M, N_values) {
  L2_errors <- numeric(length(N_values))
  
  for (j in seq_along(N_values)) {
    N <- N_values[j]
    dt <- T / N
    
    # Initialize error accumulation
    errors <- numeric(M)
    
    for (i in 1:M) {
      W <- cumsum(c(0, sqrt(dt) * rnorm(N)))
      
      S_exact <- s0 * exp((r - 0.5 * sigma^2) * T + sigma * W[N + 1])
      S_milstein <- s0
      for (k in 1:N) {
        dW <- W[k + 1] - W[k]
        S_milstein <- S_milstein + r * S_milstein * dt + sigma * S_milstein * dW + 0.5 * sigma^2 * S_milstein * (dW^2 - dt)
      }
      
      errors[i] <- (S_exact - S_milstein)^2
    }
    
    L2_errors[j] <- mean(errors)
  }
  
  return(L2_errors)
}

# Parameters
N_values <- c(10, 20, 50, 100, 200, 500, 1000)  # Different values of N

# Compute L^2 errors for the Milstein scheme
L2_errors_milstein <- compute_L2_error_milstein(s0, r, sigma, T, M, N_values)

# Plot L^2 error as a function of N
plot_data_milstein <- data.frame(
  N = N_values,
  L2_Error = L2_errors_milstein
)

p3 <- ggplot(plot_data_milstein, aes(x = N, y = L2_Error)) +
  geom_line() +
  geom_point() +
  labs(title = "L^2 Error for Milstein Scheme as a Function of N",
       x = "Number of Time Steps (N)",
       y = "L^2 Error") +
  theme_minimal()

# Save the plot
ggsave("exercise_4_8_L2_error_milstein.png", plot = p3)

# Print the L^2 errors for Milstein scheme
cat("L^2 errors for Milstein scheme:\n")
print(plot_data_milstein)

# N     L2_Error
# 1   10 1.340390e-02
# 2   20 3.489025e-03
# 3   50 5.786265e-04
# 4  100 1.329387e-04
# 5  200 3.610572e-05
# 6  500 5.354944e-06
# 7 1000 1.431723e-06