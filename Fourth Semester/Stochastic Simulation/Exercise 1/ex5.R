############################
# ex6.R 
############################
setwd("/Users/Q540900/Desktop/A.I. Master/Fourth Semester/Stochastic Simulation/Exercise 1")

# 1) Quadratic CG (from Exercise 5)
qcg <- function(d, a, c, m, x0) {
  vals <- integer(0)
  curr <- x0
  repeat {
    next_val <- (d * curr^2 + a * curr + c) %% m
    vals <- c(vals, next_val)
    curr <- next_val
    if (curr == x0) {
      break
    }
  }
  return(vals)
}

# 2) Try searching for a full-period generator
period_length <- function(d, a, m=65536, x0=1, c=1) {
  seq_ <- qcg(d, a, c, m, x0)
  length(seq_)
}

best_period <- 0
best_params <- c(NA,NA)

for (d_try in 1:100) { # loop over numbers
  for (a_try in seq(1,99,2)) { # loop over numbers
    p <- period_length(d_try, a_try)
    if (p > best_period) {
      best_period <- p
      best_params <- c(d_try, a_try)
      
      if (best_period == 65536) { # Add a breaking
        break
      }
    }
  }
  if (best_period == 65536) {
    break
  }
}

cat("Best period found in [1..100]:", best_period, 
    "with d=", best_params[1], "and a=", best_params[2], "\n")

# 3) Let's generate with that (or override with known good values)
d <- best_params[1]
a <- best_params[2]
m <- 65536
x0 <- 1
c <- 1

seq_full <- qcg(d, a, c, m, x0)
cat("Actual period length =", length(seq_full), "\n")

# 4) Check distribution with a KS test
u_vals <- seq_full / m
cat("KS test results:\n")
print( ks.test(u_vals, "punif") )

# 5) Plot 1000 pairs (u_i, u_{i+1})
png("qcg_1000_pairs.png", 800, 600)
plot(u_vals[1:999], u_vals[2:1000],
     pch=16, cex=0.5,
     main="First 1000 consecutive pairs from QCG",
     xlab="u[i]", ylab="u[i+1]")
dev.off()
cat("Plot saved as qcg_1000_pairs.png.\n")
