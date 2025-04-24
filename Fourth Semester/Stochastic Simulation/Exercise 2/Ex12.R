rnorm_bm <- function(n) {
  u1 <- runif(n) # U(0,1)
  u2 <- runif(n) # U(0,1)
  
  r  <- sqrt(-2 * log(u1))
  z1 <- r * cos(2 * pi * u2) # first
  z2 <- r * sin(2 * pi * u2) # second
  
  c(z1, z2)                 
}

set.seed(2024)
n  <- 1e5
X  <- rnorm_bm(n)

# Part B
dens <- density(X)

plot(dens, main = "Empirical density vs.  φ(x)",
     xlab = "x", ylab = "density", lwd = 2)
curve(dnorm, from = -4, to = 4, add = TRUE,
      lwd = 2, lty = 2)
legend("topright", c("kernel density", "φ(x)"),
       lwd = 2, lty = c(1, 2))

# Part C
ks <- ks.test(X, "pnorm")            
ks$statistic    
ks$p.value

