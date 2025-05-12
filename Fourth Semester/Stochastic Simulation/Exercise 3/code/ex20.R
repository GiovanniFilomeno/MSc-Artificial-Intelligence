rm(list = ls())
set.seed(123)    
lambda_fun <- function(t) 5 * sin(t / 2)^2
lambda_max <- 5 # lambda sup 
Lambda_fun <- function(t) 2.5 * (t - sin(t)) # exaplained in the pdf

simulate_nhpp_times <- function(lambda_fun, lambda_max, T) {
  times <- numeric(0)
  t     <- 0
  repeat {
    t <- t + rexp(1, rate = lambda_max)   # candidate
    if (t > T) break
    if (runif(1) <= lambda_fun(t) / lambda_max)
      times <- c(times, t) # point accepted
  }
  times
}

simulate_Nt_nhpp <- function(lambda_fun, lambda_max, t) {
  length(simulate_nhpp_times(lambda_fun, lambda_max, t))
}

# --------- Part b ---------
T_path <- 30
dt <- 0.05

path_arrivals <- simulate_nhpp_times(lambda_fun, lambda_max, T_path)
grid <- seq(0, T_path, by = dt)
Nt_values <- findInterval(grid, path_arrivals)

plot(grid, Nt_values, type = "s", lwd = 2,
     main = expression(paste("NHPP path –  ", lambda(t)==5*sin^2*(t/2),
                             ",   T = 30")),
     xlab = "t", ylab = expression(N[t]))

# --------- Part c ---------
t_c <- 5
M <- 1e4

N5 <- replicate(M, simulate_Nt_nhpp(lambda_fun, lambda_max, t_c))

## Empirical pmf
emp_pmf <- table(N5) / M
k_vals <- as.integer(names(emp_pmf))

# Theoretical pmf  (Poisson with avg Λ(5))
lambda_bar <- Lambda_fun(t_c)   # ≈ 2.5*(5 - sin5)
true_pmf <- dpois(k_vals, lambda_bar)

# Comparison graph
barplot(emp_pmf, names.arg = k_vals, col = "lightgrey",
        main = expression(paste("Empirical vs true pmf of  ", N[5])),
        ylab = "Probability")
points(seq_along(k_vals), true_pmf, pch = 19, cex = 0.9)
legend("topright",
       legend = c("Empirical", bquote(Poisson(~lambda==.(round(lambda_bar,4))))),
       pch = c(22, 19),
       pt.bg = c("lightgrey", NA),
       pt.cex = c(2, 1))

# Some checks
cat("\nCheck\n")
cat("Theoretical mean =", lambda_bar, "\n")
cat("Theoretical variance =", lambda_bar, "\n")
cat("Empirical mean =", mean(N5), "\n")
cat("Empirical variance =", var(N5),  "\n")
