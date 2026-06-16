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