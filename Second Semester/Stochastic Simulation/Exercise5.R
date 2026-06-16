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
