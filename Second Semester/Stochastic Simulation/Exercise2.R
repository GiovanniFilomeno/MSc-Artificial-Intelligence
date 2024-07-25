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
