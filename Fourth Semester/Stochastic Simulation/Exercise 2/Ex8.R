rgamma_intshape_inv <- function(M, shape, rate = 1) {
  # Sanity check
  if (shape <= 0 || rate <= 0)
    stop("Both 'shape' and 'rate' must be positive.")
  if (abs(shape - round(shape)) > .Machine$double.eps^0.5)
    stop("'shape' must be an integer for this generator.")
  
  k <- as.integer(round(shape))
  
  U   <- runif(M * k)
  mat <- matrix(-log(U) / rate, nrow = k) 
  colSums(mat)
}

# Exercise Parameters
set.seed(2024)
M  <- 1000
n  <- 5          
lam <- 0.6       

# Generating Points
g <- rgamma_intshape_inv(M, shape = n, rate = lam)

# Defining theoretical CDF and PDF
Fgamma <- function(t) pgamma(t, shape = n, rate = lam)   # built-in pgamma
fgamma <- function(t) dgamma(t, shape = n, rate = lam)

# Part B
plot(ecdf(g), do.points = FALSE,
     main = bquote(Gamma(.(n),~lambda^{-1}==.(1/lam)):~ecdf~vs.~true~CDF),
     xlab = "x", ylab = "F(x)")
curve(Fgamma, from = 0, to = max(g), add = TRUE, lwd = 2)
legend("bottomright", c("empirical CDF", "true CDF"),
       lty = c(1,1), lwd = c(1,2))

# Part C
plot(density(g), main = "Empirical density vs. true Gamma PDF",
     xlab = "x", ylab = "f(x)")
curve(fgamma, from = 0, to = max(g), add = TRUE, lwd = 2)
legend("topright", c("kernel density", "true pdf"),
       lty = c(1,1), lwd = c(1,2))
