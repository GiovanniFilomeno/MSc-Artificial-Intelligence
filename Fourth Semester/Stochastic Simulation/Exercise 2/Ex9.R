simx <- function(M) {
  replicate(M, {
    # Initialization
    s <- 0
    n <- 0
    repeat {
      s <- s + log(runif(1))   # add log(U_i)
      if (s > -3) {
        n <- n + 1             # if above threshold -> count it
      } else {
        return(n)              # threshold crossed -> stop
      }
    }
  })
}

set.seed(2024)
M <- 1e4
x <- simx(M)

## point estimates
x_bar  <- mean(x)                       
x_var  <- var(x) 

print(c(mean = x_bar, var = x_var))

k_vals <- 0:max(x)                      
p_hat  <- tabulate(x + 1, nbins = length(k_vals)) / M
p_theo <- dpois(k_vals, lambda = 3)

data.frame(
  i        = k_vals,             # value of X
  P_hat    = signif(p_hat , 4),  # P(X = i)
  P_theory = signif(p_theo, 4)   # Poisson(3)
)


p_theo <- dpois(k_vals, lambda = 3)

barplot(rbind(p_hat, p_theo), beside = TRUE, names.arg = k_vals,
        legend = c("empirical", "Poisson(3)"),
        main = "Empirical vs. Poisson(3) probabilities",
        ylab = "P(X = k)")

# Chi-squared test section
cutoff <- 7
counts_emp <- c(p_hat[1:cutoff]*M, sum(p_hat[(cutoff+1):length(p_hat)])*M)
probs_theo <- c(p_theo[1:cutoff], sum(p_theo[(cutoff+1):length(p_theo)]))
p_value_chisq <- chisq.test(counts_emp, p = probs_theo / sum(probs_theo))$p.value

# Print the p-value
print(paste("Chi-squared test p-value:", p_value_chisq))
