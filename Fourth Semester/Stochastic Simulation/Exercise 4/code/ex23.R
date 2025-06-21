rm(list = ls())
library(stats)

## Parameters 
M     <- 1e3              
t_star <- 1
X0    <- 5
mu    <- -0.05
sigma <- 0.20
ell   <- 2:10              
h_vec <- 2^(-ell)          # step sizes

## Helper function
simulate_path_and_errors <- function(h) {
  n <- as.integer(t_star / h)
  
  ## Pre-allocate error vectors
  err_EM  <- numeric(M)
  err_Mil <- numeric(M)
  
  for (m in 1:M) {
    X_EM  <- X0
    X_Mil <- X0
    W_tot <- 0 # --> accumulutate to W_{t*}
    
    ## loop over time steps
    for (k in 1:n) {
      Z   <- rnorm(1)           
      dW  <- sqrt(h) * Z
      W_tot <- W_tot + dW
      
      ## Euler–Maruyama
      X_EM  <- X_EM  + mu * X_EM  * h + sigma * X_EM  * dW
      
      ## Milstein
      X_Mil <- X_Mil + mu * X_Mil * h + sigma * X_Mil * dW +
        0.5 * sigma^2 * X_Mil * (dW^2 - h)
    }
    
    ## Exact endpoint for Brownian motion
    X_exact <- X0 * exp((mu - 0.5 * sigma^2) * t_star + sigma * W_tot)
    
    err_EM[m]  <- (X_EM  - X_exact)^2
    err_Mil[m] <- (X_Mil - X_exact)^2
  }
  
  c(RMSE_EM  = sqrt(mean(err_EM)),
    RMSE_Mil = sqrt(mean(err_Mil)))
}

## Main Monte-Carlo loop over step sizes 
res <- t( sapply(h_vec, simulate_path_and_errors) )
colnames(res) <- c("EM", "Milstein")

## log-log plot
log2_h  <- log2(h_vec)
log2_EM <- log2(res[, "EM"])
log2_Mi <- log2(res[, "Milstein"])

plot(log2_h, log2_EM,
     type = "b", pch = 17, col = "blue", lwd = 2,
     xlab = expression(log[2](h)),
     ylab = expression(log[2](RMSE(h))),
     ylim = range(c(log2_EM, log2_Mi)),
     main = "Strong mean–square convergence: GBM")

lines(log2_h, log2_Mi, type = "b", pch = 19, col = "red", lwd = 2)

legend("topleft", legend = c("EM", "Milstein"),
       col = c("blue", "red"), pch = c(17, 19), lwd = 2, bty = "n",
       inset = 0.01)


guide_line <- function(slope,
                       x_anchor = -10,      # where you want the line to cross
                       y_anchor = -6,       # the vertical position at that x
                       ...) {
  abline(a = y_anchor - slope * x_anchor,   # compute intercept  ‘a’
         b = slope, ...)                    # slope ‘b’ is what matters
}

## Reference
guide_line(0.5, x_anchor = -9, y_anchor = -6.5,
           lty = 3, col = "grey50")
guide_line(1.0, x_anchor = -9, y_anchor = -12,
           lty = 2, col = "grey70")
legend("bottomright",
       legend = c("Order 1/2", "Order 1"),
       lty    = c(3, 2),
       col    = c("grey50", "grey70"),
       bty    = "n",
       inset = 0.01)