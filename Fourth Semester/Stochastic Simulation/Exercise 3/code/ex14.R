# Part A
RW <- function(N, p = 0.5) {
  # Generating all the steps
  steps <- sample(c(-1, 1), size = N, replace = TRUE, prob = c(1 - p, p))
  c(0, cumsum(steps))
}

# Exercise parameters
set.seed(2024)
N <- 8
path <- RW(N)

plot(0:N, path, type = "o", pch = 16, col = "steelblue",
     xlab = "n", ylab = expression(X[n]),
     ylim = range(path))
grid()


# Part B
# Exercise parameters
set.seed(2024)
n<- 8
R<- 1e5
Xn- replicate(R, sum(sample(c(-1,1), n, replace = TRUE)))

k_vals<- seq(-n, n, by = 2)
pmf<- dbinom((k_vals + n)/2, size = n, prob = 0.5)

hist(Xn,
     breaks = seq(-n - 1, n + 1, by = 2),   
     freq = FALSE, 
     col = "lightgrey",
     border = "black",
     xlab = expression(X[n]),
     ylab = "Density",
     main = "")

points(k_vals, pmf / 2, pch = 19, col = "red", cex = 1.2)   # ↓ divide per 2
lines(k_vals, pmf / 2, col = "red", lwd = 2)

legend("topright", legend = c("Simulations", "Theoretical PMF"),
       pch = c(22, 19), pt.bg = c("lightgrey", "red"),
       col = c("black", "red"), bty = "n")

