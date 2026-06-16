######################
##  ex4.R
######################

###############
## (a) Fks(x): Kolmogorov CDF  F_K(x) = P(K ≤ x)
##    We use the standard series expansion for x>0:
##      F_K(x) = 1 - 2 * sum_{k=1..∞} (-1)^{k-1} exp(-2 k^2 x^2).
##    We truncate at a finite number of terms (e.g. 50).
##    Also define Fks(x) = 0 for x <= 0.

Fks <- function(x, terms = 100) {
  if (x <= 0) return(0) # as requested in the exercise
  sum_val <- 0
  for (k in seq_len(terms)) { # Added a truncation as requested in the ex.
    sum_val <- sum_val + (-1)^(k-1) * exp(-2 * (k^2) * x^2)
  }
  # The series gives 1 - 2 * sumVal
  return(1 - 2 * sum_val)
}

# Vectorize as requested
Fks <- Vectorize(Fks)

###############
## (b) Kalpha(alpha): find K_alpha such that P(K ≤ K_alpha) = 1 - alpha
##    i.e. Fks(K_alpha) = 1 - alpha.
##    We solve numerically for K_alpha > 0 using uniroot.

Kalpha <- function(alpha) {
  if (alpha <= 0 || alpha >= 1) {
    stop("alpha must be in (0,1).")
  }
  target <- 1 - alpha   # we want Fks(K_alpha) = target
  f <- function(x) Fks(x) - target
  
  # Bracking the root to avoid unstability
  sol <- uniroot(f, interval = c(1e-5, 5)) # using uniroot as suggested
  
  return(sol$root)
}

###############
## (c) Dno(A): Kolmogorov distance for sample A under H0: Uniform(0,1)
##    D_n = sup_x | F_n(x) - x |.
##    A quick formula uses the sorted sample.

Dno <- function(A) {
  A_sorted <- sort(A)
  n <- length(A)
  
  # D+ = max_{i} [i/n - X_(i)]
  Dplus  <- max( (seq_len(n))/n - A_sorted )
  # D- = max_{i} [X_(i) - (i-1)/n ]
  Dminus <- max( A_sorted - (seq_len(n)-1)/n )
  
  D <- max(Dplus, Dminus)
  return(D)
}

###############
## (d) pval(A): the p-value for the one-sample KS test
##    For large n, we use the limiting distribution:
##       p-value = 1 - F_K( sqrt(n)* D_n ).
##    That is, p = 1 - Fks( sqrt(n)*D ).
##    (This is the usual approximation that `ks.test` also uses.)

pval <- function(A) {
  n <- length(A)
  d <- Dno(A)
  stat <- sqrt(n)*d
  return(1 - Fks(stat))
}

###############
## (e) ourkstest(A, alpha): perform Kolmogorov-Smirnov test at level alpha.
##    Return both the p-value and the Kolmogorov distance.

ourkstest <- function(A, alpha) {
  # Computing D using 4.c function
  D <- Dno(A)
  # Computing p-value using 4.d function
  p <- pval(A)
  # Making decision: p < alpha --> reject. 
  decision <- if (p < alpha) "Reject H0" else "Do not reject H0"
  
  result <- list(
    D = D, # Requested by ex.
    p.value = p, # Requested by ex.
    alpha = alpha, # not asked, but useful for the print
    decision = decision # not asked, but useful for reject/not reject test
  )
  return(result)
}

###############
# 5.f
set.seed(11)
Asim <- runif(50)

# Try alpha = 0.01 and alpha = 0.05 as requested
res1 <- ourkstest(Asim, 0.01)
res2 <- ourkstest(Asim, 0.05)

cat("\nourkstest result, alpha=0.01:\n")
print(res1)

cat("\nourkstest result, alpha=0.05:\n")
print(res2)

# Compare with ks.test:
cat("\nCompare with built-in ks.test:\n")
print( ks.test(Asim, "punif") )
