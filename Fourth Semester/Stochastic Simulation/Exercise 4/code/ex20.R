## Part A: Wiener process
simulate_wiener_drift <- function(mu     = 0,
                                  sigma  = 1,
                                  T_end  = 1,
                                  h      = 1e-3)
{
  n  <- ceiling(T_end / h)            # number of steps
  t  <- seq(0, T_end, length.out = n + 1)
  
  ## Brownian increments
  dW <- sqrt(h) * rnorm(n)
  W  <- c(0, cumsum(dW))              
  
  X  <- mu * t + sigma * W
  
  data.frame(t = t, X = X, W = W)
}

## Part B: 5 paths 
set.seed(42)                          # reproducibility

h_small <- 1e-3
paths_b <- replicate(
  5,
  simulate_wiener_drift(mu = 5, sigma = 1, T_end = 1, h = h_small)$X,
  simplify = "matrix"
)

time_b <- seq(0, 1, by = h_small)

## Plot
my_colors <- c("black", "blue", "green", "cyan", "magenta")
matplot(time_b, paths_b, type = "l", lty = 1, col = my_colors,
        xlab = "time  t", ylab = expression(X[t]),
        main = expression("Five paths of   X"[t]==5*t+W[t]))
lines(time_b, 5 * time_b, lwd = 2, col = "red")
legend("topleft",
       legend = c(paste("Sample path", 1:5), expression(E[X[t]]==5*t)),
       lwd = c(rep(1, 5), 2),
       col = c(my_colors, "red"),
       bty = "n")

## Part C: 5 paths changing mu
mu_set   <- c(0, -1, 1, -5, 5)
T_long   <- 10
paths_c  <- sapply(
  mu_set,
  function(mu) simulate_wiener_drift(mu, sigma = 1,
                                     T_end = T_long, h = h_small)$X
)

time_c <- seq(0, T_long, by = h_small)

matplot(time_c, paths_c, type = "l", lty = 1,
        xlab = "time  t", ylab = expression(X[t]),
        main = expression("Wiener processes with different drifts,  σ=1"))
legend("topleft",
       legend = paste0("μ = ", sprintf("% 2d", mu_set)),
       col = 1:length(mu_set), lty = 1, bty = "n")

## Part D: 5 paths changing sigma
sigma_set <- c(0, 1, 5, 10)      # the volatilities we want
T_long    <- 10                  # same horizon as part C
h_small   <- 1e-3                # time–step Δt

## Generate one standard Brownian path 
W_path <- simulate_wiener_drift(mu = 0, sigma = 1,
                                T_end = T_long, h = h_small)$W

paths_d <- outer(W_path, sigma_set)
time_d  <- seq(0, T_long, by = h_small)

## Plot
matplot(time_d, paths_d, type = "l", lty = 1,
        xlab = "time  t", ylab = expression(X[t]),
        main = expression("Brownian motion with μ=0 and various σ"))
legend("topleft",
       legend = paste0("σ = ", sigma_set),
       col = 1:length(sigma_set), lty = 1, bty = "n")