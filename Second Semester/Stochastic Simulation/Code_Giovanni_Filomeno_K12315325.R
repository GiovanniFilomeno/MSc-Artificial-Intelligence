### -------------------------------------------
### Exercise 1
### Topic 1: Pseudo-random number generator RANDU
### -------------------------------------------

##### Part 1.1: Implement a RANDU generator
generate_randu <- function(M) {
  a <- 65539 # costant
  m <- 2^31 # modulo
  x <- numeric(M)
  x[1] <- 1 # seed = 1 as requested in the exercise
  for (i in 2:M) {
    x[i] <- (a * x[i-1]) %% m
  }
  u <- x / m
  return(u)
}

M <- 10^4 # used in exercise 1.4
randu_numbers <- generate_randu(M)

##### Part 1.2: Illustrate RANDU in 2D

png("exercise_1_2_RANDU_2D.png")
plot(randu_numbers[1:(M-1)], randu_numbers[2:M], 
     main = "RANDU in two dimensions", 
     xlab = expression(u[i-1]), ylab = expression(u[i]), # label of axis
     pch = 19, col = "blue", cex = 0.1) # scaling the points
dev.off()

##### Part 1.3: Extended the visualisation to 3 dimensions
if (!requireNamespace("scatterplot3d", quietly = TRUE)) {
  install.packages("scatterplot3d")
}

if (!requireNamespace("rgl", quietly = TRUE)) {
  install.packages("rgl")
}

library(scatterplot3d)
png("exercise_1_3_RANDU_3D.png")
scatterplot3d(randu_numbers[1:(M-2)], randu_numbers[2:(M-1)], randu_numbers[3:M], 
              main = "RANDU in three dimensions", 
              xlab = expression(u[i]), 
              ylab = expression(u[i-1]), # label of axis 
              zlab = expression(u[i-2]), 
              pch = 19, color = "blue", cex.symbols = 0.1) # scaling the points
dev.off()

library(rgl)

# Creating 3D interactive graph
plot3d(randu_numbers[1:(M-2)], 
       randu_numbers[2:(M-1)], # randu coordinates of points
       randu_numbers[3:M],
       col = "blue", size = 3, 
       xlab = expression(u[i]), 
       ylab = expression(u[i-1]), # label of axis
       zlab = expression(u[i-2]))


##### Part 1.4: Approximate the number pi

approximate_pi <- function(u) {
  M <- length(u) / 2 # splitting half of length
  x <- u[1:M] # first half
  y <- u[(M+1):(2*M)] # second half
  inside_circle <- sum(x^2 + y^2 <= 1) #compute is inside the circle
  pi_approx <- (inside_circle / M) * 4 # calculate pi
  return(pi_approx)
}

pi_approximation <- approximate_pi(randu_numbers)
print(pi_approximation) # result 3.1496

# Visualization
x <- randu_numbers[1:(M/2)]
y <- randu_numbers[(M/2+1):M]
inside_circle <- x^2 + y^2 <= 1

png("exercise_1_4_pi_approximation.png")
plot(x, y, col = ifelse(inside_circle, "blue", "red"), 
     main = "Monte Carlo Approximation of Pi", 
     xlab = "x", ylab = "y", pch = 19, cex = 0.5)
legend("topright", legend = c("Inside Circle", "Outside Circle"), 
       col = c("blue", "red"), pch = 19)
dev.off()

### -------------------------------------------
### Exercise 2
### Topic 2: Acceptance-rejection method and Box-Muller algorithm
### -------------------------------------------

# Define the target probability density function
f <- function(x) {
  (2 / sqrt(2 * pi)) * exp(-x^2 / 2)
}

# Define the exercise density function g(x)
g <- function(x) {
  exp(-x)
}

# Determine the constant c as the maximum of f(x) / g(x)
c <- sqrt(2 * exp(1) / pi)

##### Part 2.1: Implement the acceptance-rejection method

set.seed(123)  # Random seed for reproducibility
M <- 10^4  # Number of realizations from exercise
samples <- numeric(M)  # Initialize the vector
count <- 0  # Counter for the number of accepted samples
total_trials <- 0 # Counter for the total number of trials

# Perform acceptance-rejection 
while (count < M) {
  Y <- rexp(1)  # Sample from the proposal distribution g(x)
  U <- runif(1)  # Sample from uniform random number 0-1
  total_trials <- total_trials + 1  # Increment the total number of trials
  # Check if it is less or equal than the acceptance ratio
  if (U <= f(Y) / (c * g(Y))) {
    count <- count + 1 # Counter for accepted samples
    samples[count] <- Y # Store accepted sample
  }
}

# Display the first few accepted samples
head(samples)
# 0.8434573 1.3290549 0.1028700 0.3142273 1.4154478 0.3856068

##### Part 2.2: Determine the acceptance probability

# Theoretical acceptance probability
acceptance_probability_theoretical <- 1 / c
print(acceptance_probability_theoretical)
# 0.7601735

# Observed acceptance probability
acceptance_probability_observed <- M / total_trials
print(acceptance_probability_observed)
# 0.7634753

##### Part 2.3: Implement the Box-Muller algorithm

set.seed(123)  # For reproducibility
M <- 10^4  # Number of realizations
Z1 <- numeric(M) # Initialize vector for first normal samples
Z2 <- numeric(M) # Initialize vector for second normal samples

# Generating normal samples using Box-Muller transform
for (i in 1:(M/2)) {
  U1 <- runif(1) # Generate first uniform random number
  U2 <- runif(1) # Generate second uniform random number
  
  # Apply Box-Muller transform 
  Z1[i] <- sqrt(-2 * log(U1)) * cos(2 * pi * U2) 
  Z2[i] <- sqrt(-2 * log(U1)) * sin(2 * pi * U2)
}

Z <- c(Z1, Z2)  # Combine Z1 and Z2
head(Z) # Display results
# 0.3763186  0.9919799  0.3361111  0.8815332 -1.0507947 -0.2843938

##### Part 2.4: Compare empirical pdfs

# Absolute values of the Box-Muller samples
Z_abs <- abs(Z)

# Plotting area to have two plots in one column
par(mfrow = c(2, 1))

# Histogram for the acceptance-rejection samples
png("exercise_2_4_empirical_pdf_AR.png")
hist(samples, breaks = 50, probability = TRUE, main = "Empirical PDF - Acceptance-Rejection", xlab = "x", col = "lightblue")
curve(f, add = TRUE, col = "red", lwd = 2)
dev.off() # close png device

# Histogram for the absolute values of the Box-Muller samples
png("exercise_2_4_empirical_pdf_BM.png")
hist(Z_abs, breaks = 50, probability = TRUE, main = "Empirical PDF - Box-Muller (|Z|)", xlab = "|Z|", col = "lightgreen")
curve(f, add = TRUE, col = "red", lwd = 2)
dev.off() # close png device

### -------------------------------------------
### Exercise 3
### Topic 3: Markov chain
### -------------------------------------------

# Define the transition matrix P as defined in the exercise
# It represent the transition probabilities between the 5 states
P <- matrix(c(0.93, 0.07, 0,    0,    0,
              0.05, 0.8,  0.1,  0.05, 0,
              0,    0.15, 0.8,  0.05, 0,
              0,    0,    0.05, 0.8,  0.15,
              0,    0,    0,    0,    1), 
            nrow = 5, byrow = TRUE)

# State names
states <- c("non-infected", "infected", "hospitalized", "intensive care unit", "dead")

##### Part 3.1: Produce a corresponding transition diagram

# Check if the necessary packages are installed and load them
if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
  install.packages("DiagrammeR")
}
library(DiagrammeR)
if (!requireNamespace("DiagrammeRsvg", quietly = TRUE)) {
  install.packages("DiagrammeRsvg")
}
library(DiagrammeRsvg)
if (!requireNamespace("rsvg", quietly = TRUE)) {
  install.packages("rsvg")
}
library(rsvg)

# Define transition diagram
graph <- grViz("
digraph markov_chain {
  rankdir=LR;
  node [shape = circle];
  1 [label = 'non-infected'];
  2 [label = 'infected'];
  3 [label = 'hospitalized'];
  4 [label = 'intensive care unit'];
  5 [label = 'dead'];
  
  1 -> 1 [label = '0.93'];
  1 -> 2 [label = '0.07'];
  
  2 -> 1 [label = '0.05'];
  2 -> 2 [label = '0.80'];
  2 -> 3 [label = '0.10'];
  2 -> 4 [label = '0.05'];
  
  3 -> 2 [label = '0.15'];
  3 -> 3 [label = '0.80'];
  3 -> 4 [label = '0.05'];
  
  4 -> 3 [label = '0.05'];
  4 -> 4 [label = '0.80'];
  4 -> 5 [label = '0.15'];
  
  5 -> 5 [label = '1'];
}
")

# Save the diagram as SVG
svg_filename <- "exercise_3_1_transition_diagram.svg"
svg_content <- export_svg(graph)

# Save the SVG content to a file
write(svg_content, file = svg_filename)

# Convert the SVG to PNG
png_filename <- "exercise_3_1_transition_diagram.png"
rsvg_png(svg_filename, png_filename)


##### Part 3.2: Implement an algorithm to generate a path of the Markov chain

set.seed(123)  # For reproducibility

# Generate the path of the Markov chain
generate_path <- function(P, X0, N) {
  path <- numeric(N + 1) # Initialize vector
  path[1] <- X0 # Set initial condition
  for (i in 2:(N + 1)) {
    path[i] <- sample(1:5, size = 1, prob = P[path[i - 1], ]) # sample next state
  }
  return(path)
}

X0 <- 1  # Initial state: non-infected
N <- 30  # Number of time steps

# Generate a path of the Markov chain starting from the non-infected for 30 time step
path <- generate_path(P, X0, N)

# Display the path
path_states <- states[path]
path_states

# [1] "non-infected"        "non-infected"        "non-infected"        "non-infected"        "non-infected"        "infected"            "infected"           
# [8] "infected"            "hospitalized"        "hospitalized"        "hospitalized"        "intensive care unit" "intensive care unit" "intensive care unit"
# [15] "intensive care unit" "intensive care unit" "dead"                "dead"                "dead"                "dead"                "dead"               
# [22] "dead"                "dead"                "dead"                "dead"                "dead"                "dead"                "dead"               
# [29] "dead"                "dead"                "dead"  

# Plot the path
png("exercise_3_2_markov_chain_path.png")
plot(0:N, path, type = "s", main = "Markov Chain Path", xlab = "Time Step", ylab = "State", xaxt = "n", yaxt = "n")
axis(1, at = 0:N)
axis(2, at = 1:5, labels = states)
dev.off()

##### Part 3.3: Consider the random variable τ

set.seed(123)  # For reproducibility

# Estimate the life expectancy
estimate_life_expectancy <- function(P, X0, M) {
  tau <- numeric(M) # Initialize the vector for the time to absorption
  for (i in 1:M) {
    n <- 0 # Initialize time counter
    state <- X0 # Set initial state
    while (state != 5) {  # Until absorbing state (dead) is reached
      state <- sample(1:5, size = 1, prob = P[state, ]) # Sample the next state
      n <- n + 1 # Increment time counter
    }
    tau[i] <- n
  }
  return(mean(tau))
}

M <- 10^4  # Number of realizations
life_expectancy <- numeric(4) # Initialize vector for life expectancy
for (i in 1:4) {
  life_expectancy[i] <- estimate_life_expectancy(P, i, M)
}

# Display the life expectancy for states 1 to 4
life_expectancy
# 59.4481 45.0570 42.8106 16.2120

##### Part 3.4: Compare with theoretical expectations

# Extract transient state (submatrix) Q
Q <- P[1:4, 1:4]

# Identity matrix I
I <- diag(4)

# Fundamental matrix F
F <- solve(I - Q)

# Vector of ones
ones <- rep(1, 4)

# Expected time to absorption (life expectancy)
F1 <- F %*% ones

# Display the theoretical life expectancy
F1
# 59.52381
# 45.23810
# 42.85714
# 15.71429

# Comparison
comparison <- data.frame(State = states[1:4], 
                         Simulated = life_expectancy, 
                         Theoretical = F1)
print(comparison)

# State - Simulated - Theoretical
# non infected - 59.4481 - 59.52381
# infected - 45.0570 - 45.23810
# hospitalized - 42.8106 - 42.85714
# intensive care unit - 16.2120 - 15.71429

# Save the comparison as PNG
if (!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra")
}
if (!requireNamespace("grid", quietly = TRUE)) {
  install.packages("grid")
}
library(gridExtra)
library(grid)
png("exercise_3_4_comparison.png")
grid.table(comparison)
dev.off()

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

### -------------------------------------------
### Exercise 5
### Topic 5: Monte Carlo Integration
### -------------------------------------------

# Load necessary packages
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
library(ggplot2)

set.seed(123)

##### Part 5.1: True value of theta (from Wolfram Alpha)
theta_true <- -0.132784

# Define the function f(x) = x^2 * sin(pi * x) * exp(-x^2 / 2)
f <- function(x) {
  x^2 * sin(pi * x) * exp(-x^2 / 2)
}

# Monte Carlo integration using Y ~ Exp(1/2)
monte_carlo_exp <- function(n) {
  Y <- rexp(n, rate = 1/2)
  g <- f(Y) / (1/2 * exp(-Y/2))
  mean(g)
}

# Monte Carlo integration using Y ~ Chi-squared(6)
monte_carlo_chi <- function(n) {
  Y <- rchisq(n, df = 6)
  g <- f(Y) / (1/(2^(3) * gamma(3)) * Y^(3-1) * exp(-Y/2))
  mean(g)
}

# Simulate the estimations for different n
simulate_estimations <- function(n_values, estimator) {
  estimates <- numeric(length(n_values))
  for (i in 1:length(n_values)) {
    estimates[i] <- estimator(n_values[i])
  }
  return(estimates)
}

# Values of n to consider
n_values <- seq(100, 10000, by = 100)

# Simulate the estimations
estimates_exp <- simulate_estimations(n_values, monte_carlo_exp)
estimates_chi <- simulate_estimations(n_values, monte_carlo_chi)

# Create a data frame for plotting
data_exp <- data.frame(n = n_values, estimate = estimates_exp, method = "Exp(1/2)")
data_chi <- data.frame(n = n_values, estimate = estimates_chi, method = "Chi-squared(6)")

# Plot the estimates for Exp(1/2)
p_exp <- ggplot(data_exp, aes(x = n, y = estimate)) +
  geom_line(color = "blue") +
  geom_hline(yintercept = theta_true, linetype = "dashed", color = "red") +
  labs(x = "Number of samples (n)",
       y = "Estimate of theta") +
  theme_minimal()

# Save the plot
ggsave("exercise_5_3_estimates_exp.png", plot = p_exp)

# Plot the estimates for Chi-squared(6)
p_chi <- ggplot(data_chi, aes(x = n, y = estimate)) +
  geom_line(color = "green") +
  geom_hline(yintercept = theta_true, linetype = "dashed", color = "red") +
  labs(x = "Number of samples (n)",
       y = "Estimate of theta") +
  theme_minimal()

# Save the plot
ggsave("exercise_5_5_estimates_chi.png", plot = p_chi)

##### Part 5.5: Single estimation with n = 10^4
n_single <- 10^4
single_estimate_exp <- monte_carlo_exp(n_single)
single_estimate_chi <- monte_carlo_chi(n_single)

cat("Single estimate (Exp(1/2)):", single_estimate_exp, "⁄n")
# -0.1205654
cat("Single Estimate (Chi-squared(6)):", single_estimate_chi, "⁄n")
# -0.1293532

##### Part 5.6: Compare estimators for n = 10^4
n <- 10^4
M <- 100

# Simulate M realizations of the estimators
simulate_realizations <- function(M, n, estimator) {
  realizations <- numeric(M)
  for (i in 1:M) {
    realizations[i] <- estimator(n)
  }
  return(realizations)
}

realizations_exp <- simulate_realizations(M, n, monte_carlo_exp)
realizations_chi <- simulate_realizations(M, n, monte_carlo_chi)

# Compare the sample means and variances
mean_exp <- mean(realizations_exp)
var_exp <- var(realizations_exp)

mean_chi <- mean(realizations_chi)
var_chi <- var(realizations_chi)

cat("Mean (Exp(1/2)):", mean_exp, "\n")
# Mean (Exp(1/2)): -0.131227 
cat("Variance (Exp(1/2)):", var_exp, "\n")
# Variance (Exp(1/2)): 0.0001440485
cat("Mean (Chi-squared(6)):", mean_chi, "\n")
# Mean (Chi-squared(6)): -0.1316119 
cat("Variance (Chi-squared(6)):", var_chi, "\n")
# Variance (Chi-squared(6)): 0.0007547391 

# Plot the distribution of the estimators
data_realizations <- data.frame(
  value = c(realizations_exp, realizations_chi),
  method = rep(c("Exp(1/2)", "Chi-squared(6)"), each = M)
)

p2 <- ggplot(data_realizations, aes(x = value, fill = method)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
  geom_vline(xintercept = theta_true, linetype = "dashed", color = "red") +
  labs(title = "Distribution of Monte Carlo Estimators",
       x = "Estimate of theta",
       y = "Frequency") +
  theme_minimal()

# Save the plot
ggsave("exercise_5_6_distribution.png", plot = p2)
