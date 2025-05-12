rm(list = ls())
set.seed(123)    

lambda <- 3           
M <- 1e4      

## Copy-paste from ex18
simulate_poisson_times <- function(lambda, T) {
  times <- c()
  t <- 0
  repeat {
    t <- t + rexp(1, rate = lambda) 
    if (t > T) break           
    times <- c(times, t)
  }
  times
}

# Simulate Nt, counting the arrival times <= t
simulate_Nt <- function(lambda, t) {
  arrivals <- simulate_poisson_times(lambda, t)
  length(arrivals)
}

# ------ Part c ------

t_c <- 2  
N2 <- replicate(M, simulate_Nt(lambda, t_c))

# Empiric pmf
emp_pmf_N2 <- table(N2) / M

# Theoretical pmf
k_vals <- as.integer(names(emp_pmf_N2))
true_pmf_N2 <- dpois(k_vals, lambda * t_c)

# Picture
barplot(emp_pmf_N2,
        names.arg = k_vals,
        col  = "lightgrey",
        main = expression(paste("Empirical vs true pmf of  ", N[2])),
        ylab = "Probability")
points(seq_along(k_vals), true_pmf_N2, pch = 19, cex = 0.9)
legend("topright",
       legend = c("Empirical", "Poisson(λ·2)"),
       pch = c(22, 19),
       pt.bg = c("lightgrey", NA),
       pt.cex = c(2, 1))


# ------ Part d ------
# Summing 4 expected exponential
T4 <- replicate(M, sum(rexp(4, rate = lambda))) 

# theoretic pdf
shape_T4 <- 4
rate_T4 <- lambda

# Picture
hist(T4, breaks = 40, freq = FALSE, col = "lightgrey",
     main = expression(paste("Empirical pdf of  ", T[4],
                             "  vs  Gamma(4, λ = 3)")),
     xlab = expression(t))
curve(dgamma(x, shape = shape_T4, rate = rate_T4),
      add = TRUE, lwd = 2)

# Comparison
cat("\n==== Summary checks ====\n")
cat("N2:  mean =", mean(N2),
    "  (theoretical =", lambda * t_c, ")\n")
cat("     var  =", var(N2),
    "  (theoretical =", lambda * t_c, ")\n\n")

cat("T4:  mean =", mean(T4),
    "  (theoretical =", shape_T4 / rate_T4, ")\n")
cat("     var  =", var(T4),
    "  (theoretical =", shape_T4 / rate_T4^2, ")\n")
