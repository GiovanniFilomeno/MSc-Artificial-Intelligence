## Parameters
T_end  <- 1          # horizon
h      <- 1e-3       # step size Δt
n      <- ceiling(T_end / h)         
t_grid <- seq(0, T_end, by = h)     

X0     <- 5
mu     <- 1
sigma  <- 0.4

set.seed(123)                       
dW <- rnorm(n, mean = 0, sd = sqrt(h))  
W  <- c(0, cumsum(dW))                

## Euler–Maruyama
X_EM <- numeric(n + 1)
X_EM[1] <- X0
for (k in 1:n) {
  X_EM[k + 1] <- X_EM[k] +
    mu    * X_EM[k] * h +
    sigma * X_EM[k] * dW[k]
}

## Milstein
X_Mil <- numeric(n + 1)
X_Mil[1] <- X0
for (k in 1:n) {
  X_Mil[k + 1] <- X_Mil[k] +
    mu    * X_Mil[k] * h +
    sigma * X_Mil[k] * dW[k] +
    0.5 * sigma^2 * X_Mil[k] 
    * (dW[k]^2 - h)   # Milstein term
}

## Exact solution
X_exact <- X0 * exp((mu - 0.5 * sigma^2) * t_grid + sigma * W)

## Plot
par(mfrow = c(2, 1), mar = c(4, 4, 3, 2) + .1)

## Euler–Maruyama
plot(t_grid, X_EM, type = "l", lwd = 1.5,
     main = "Euler–Maruyama approximation of GBM\n(T = 1, h = 1e-3)",
     xlab = "time  t", ylab = expression(X[t]))
lines(t_grid, X_exact, col = "red", lwd = 1, lty = 2)
legend("topleft",
       legend = c("Euler–Maruyama", "Exact"),
       col = c("black", "red"), lwd = c(1.5, 1), lty = c(1, 2), bty = "n")

## Milstein
plot(t_grid, X_Mil, type = "l", lwd = 1.5,
     main = "Milstein approximation of GBM\n(T = 1, h = 1e-3)",
     xlab = "time  t", ylab = expression(X[t]))
lines(t_grid, X_exact, col = "red", lwd = 1, lty = 2)
legend("topleft",
       legend = c("Milstein", "Exact"),
       col = c("black", "red"), lwd = c(1.5, 1), lty = c(1, 2), bty = "n")
