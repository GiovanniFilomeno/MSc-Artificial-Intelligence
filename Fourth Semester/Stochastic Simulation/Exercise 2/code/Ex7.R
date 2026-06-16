set.seed(2024)
M     <- 1000
alpha <- 2
beta  <- 4

rweibull_inv <- function(n, alpha, beta) {
  U <- runif(n) # Suggested as Hint
  (-log(U) / alpha)^(1 / beta)
}

x  <- rweibull_inv(M, alpha, beta)
F  <- function(t) 1 - exp(-alpha * t^beta)        # true CDF

plot(ecdf(x), do.points = FALSE, main = "Weibull(α=2, β=4): ecdf vs. true CDF",
     xlab = "x", ylab = "F(x)")
curve(F, from = 0, to = max(x), add = TRUE, lwd = 2)
legend("bottomright", legend = c("empirical CDF", "true CDF"),
       lty = c(1,1), lwd = c(1,2), col = c("black","black"))

set.seed(456)
M     <- 1000
lambda <- 0.2

x_exp <- rweibull_inv(M, alpha = lambda, beta = 1)
Fexp  <- function(t) 1 - exp(-lambda * t)

plot(ecdf(x_exp), do.points = FALSE,
     main = "Exponential(λ = 0.2): ecdf vs. true CDF",
     xlab = "x", ylab = "F(x)")
curve(Fexp, from = 0, to = max(x_exp), add = TRUE, lwd = 2)
legend("bottomright", legend = c("empirical CDF", "true CDF"),
       lty = c(1,1), lwd = c(1,2), col = c("black","black"))
