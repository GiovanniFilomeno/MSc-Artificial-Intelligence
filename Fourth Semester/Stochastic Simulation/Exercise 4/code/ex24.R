## Part A: find theta
f <- function(x) (cos(50 * x) + sin(20 * x))^2        # integrand
theta_det <- integrate(f, lower = 0, upper = 2,
                       rel.tol = 1e-12)$value

cat(sprintf("Deterministic estimate  θ ≈ %.10f\n", theta_det))

## Part B: Use Monte-Carlo
set.seed(1)                                  
n_MC   <- 1e4
Y      <- runif(n_MC, 0, 2)     # Y ~ U(0,2)
theta_MC <- 2 * mean(f(Y))                          
cat(sprintf("MC estimate with n = 10^4 :  θ̂ ≈ %.10f\n", theta_MC))

## Part C: Plot
max_n      <- 1e4
Y_stream   <- runif(max_n, 0, 2)                     # one long stream
vals       <- f(Y_stream)
theta_tail <- 2 * cumsum(vals) / seq_along(vals)     # cumulative MC estimates
n_grid     <- seq_len(max_n)

plot(n_grid, theta_tail, type = "l", log = "x",
     xlab = "sample size  n  (log scale)",
     ylab = expression(hat(theta)[n]),
     main = expression("MC convergence for  θ  = ∫[0]^2 (cos 50x + sin 20x)^2 dx"))
abline(h = theta_det, col = "red", lwd = 2)          # deterministic reference
legend("topright",
       legend = c("MC running estimate",
                  bquote("deterministic"~~hat(theta)==.(format(theta_det, digits = 6)))),
       lwd = c(1, 2), col = c("black", "red"), bty = "n")
