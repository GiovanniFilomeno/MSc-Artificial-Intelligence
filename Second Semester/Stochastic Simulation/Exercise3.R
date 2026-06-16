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