## Part A: geometric brownian motion
simulate_gbm_exact <- function(mu     = 0,
                               sigma  = 1,
                               X0     = 1,
                               T_end  = 1,
                               h      = 1e-3)
{
  n  <- ceiling(T_end / h)            # steps
  t  <- seq(0, T_end, length.out = n + 1)
  
  ## Brownian increments 
  dW <- sqrt(h) * rnorm(n)
  W  <- c(0, cumsum(dW)) 
  
  ## Exact GBM path
  X  <- X0 * exp((mu - 0.5 * sigma^2) * t + sigma * W)
  
  data.frame(t = t, X = X)
}

## Part B: 10 paths
set.seed(2024)              

params <- list(mu = 1, sigma = 0.4, X0 = 5, T_end = 1, h = 1e-3)

## Generate 10 independent paths
paths <- replicate(
  10,
  do.call(simulate_gbm_exact, params)$X,
  simplify = "matrix"
)

time <- seq(0, params$T_end, by = params$h)

## Plot all paths
matplot(time, paths, type = "l", lty = 1,
        xlab = "time  t", ylab = expression(X[t]),
        main = bquote("10 exact GBM paths,  "~X[0]==.(params$X0)*","~
                        mu==.(params$mu)*","~sigma==.(params$sigma)))

## Add theoretical mean  E[X(t)] = X0 * exp(μ t)
lines(time, params$X0 * exp(params$mu * time), lwd = 2, col = "red", lty = 2)

legend("topleft",
       legend = c("sample paths (continue)",
                  bquote(E[X[t]] == .(params$X0)*e^{.(params$mu)*t} ~ "(dashed)")),
       lwd = c(1,2), col = c("black", "red"), lty = c(1, 2), bty = "n")

