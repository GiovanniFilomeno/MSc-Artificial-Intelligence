r_fx <- function(M) {
  out <- numeric(M)
  i   <- 1
  while (i <= M) {
    y <- runif(1)                 
    u <- runif(1)                 
    if (u <= exp(y - 1)) {        
      out[i] <- y
      i <- i + 1
    }
  }
  out
}

# Exercise Parameters
set.seed(2024)
M <- 1e5
x <- r_fx(M)

# Empirical Parameters
mean_emp <- mean(x)
var_emp  <- var(x)

# Theoretical Parameters
mu  <- 1 / (exp(1) - 1) # --> E[X]
mu2 <- (exp(1) - 2) / (exp(1) - 1) # --> E[X^2]
var_theo <- mu2 - mu^2 # --> Var(X)

# Table
data.frame(
  quantity    = c("mean", "variance"),
  empirical   = c(mean_emp, var_emp),
  theoretical = c(mu,       var_theo)
)
