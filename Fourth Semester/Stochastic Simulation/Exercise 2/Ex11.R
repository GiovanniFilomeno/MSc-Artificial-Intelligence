# Target PDF
f <- function(x) 20 * x * (1 - x)^3 # 0 < x < 1
c <- 135 / 64 # Found in the report

r_f <- function(M) {
  out    <- numeric(M)
  i      <- 1
  trials <- 0L
  while (i <= M) {
    y <- runif(1)           # proposal from g
    u <- runif(1)           # uniform for accept / reject
    trials <- trials + 1L
    if (u <= f(y) / c) {    # accept
      out[i] <- y
      i <- i + 1
    }
  }
  attr(out, "acc_rate") <- M / trials
  out
}

set.seed(2024)
M <- 1e5
x <- r_f(M)
acc_sim <- attr(x, "acc_rate")        # empirical acceptance prob

cat("Empirical acceptance probability :", round(acc_sim, 4), "\n")
cat("Theoretical acceptance probability:", round(64/135, 4), "\n")

# Density of the function
dens <- density(x, from = 0, to = 1)

# Plot
plot(dens, main = "Empirical density vs. true pdf  f(x) = 20 x (1-x)^3",
     xlab = "x", ylab = "density", lwd = 2)
curve(f, from = 0, to = 1, add = TRUE, lwd = 2, lty = 2)
legend("topright", legend = c("kernel density", "true pdf"),
       lwd = 2, lty = c(1, 2))
