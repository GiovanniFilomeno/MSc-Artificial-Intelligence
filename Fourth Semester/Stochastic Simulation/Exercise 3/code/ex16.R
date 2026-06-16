MC <- function(N, P, x0) {
  S <- nrow(P) 
  path <- integer(N + 1)
  path[1] <- x0
  for (n in 1:N)
    path[n + 1] <- sample(1:S, 1, prob = P[path[n], ])
  path
}

# Transition matrix
P <- matrix(c(0,0,1,0,0,
              1,0,0,0,0,
              0,0.5,0,0.5,0,
              0,0,0.5,0,0.5,
              0,0,0,1,0),
            nrow = 5, byrow = TRUE)

set.seed(2024)
N <- 20
path <- MC(N, P, 1)

plot(0:N, path, type = "o", pch = 16, col = "steelblue",
     ylim = c(1, 5), yaxt = "n", xlab = "n", ylab = expression(X[n]))
axis(2, at = 1:5)
grid(nx = NA, ny = NULL, lty = 2)
