## Parameters
lambda <- 3        # arrival-rate λ
T_tot  <- 10       # time horizon for the sample path (part b)
M      <- 1e4      # number of Monte-Carlo replications (parts c & d)

## Part A: single Poisson-process path
simulate_poisson_path <- function(lambda, T_end) {
  t  <- 0                       # current time
  N  <- 0                       # current count
  ts <- c(0)                    # times at which N jumps
  Ns <- c(0)                    # values of N just after each jump
  
  ## keep generating exponential waiting times until we pass T_end
  while (TRUE) {
    w <- rexp(1, rate = lambda) 
    t <- t + w
    if (t > T_end) break
    N <- N + 1
    ts <- c(ts, t)
    Ns <- c(Ns, N)
  }
  
  ts <- c(ts, T_end)
  Ns <- c(Ns, N)
  
  data.frame(time = ts, count = Ns)
}

## Part B: plot the path for λ = 3, T = 10
set.seed(42)
path <- simulate_poisson_path(lambda, T_tot)

plot(path$time, path$count,
     type = "s",               
     xlab = "time t",
     ylab = expression(N[t]),
     main = bquote("Poisson path,  "~lambda==.(lambda)*",  T="*.(T_tot)))
grid()

## Part C: N_2 empirical pmf vs. true pmf
sim_N2 <- rpois(M, lambda * 2)

## empirical pmf
pmf_hat <- table(sim_N2) / M

## plot empirical pmf
barplot(pmf_hat,
        beside = TRUE,
        names.arg = names(pmf_hat),
        xlab = expression(k),
        ylab = "Probability",
        main = expression("Empirical pmf of  N"[2]))

## overlay the true pmf
k_vals <- as.integer(names(pmf_hat))
points(1:length(k_vals),
       dpois(k_vals, lambda * 2),
       pch = 16, col = "red")
legend("topright", legend = c("Empirical", "True Poisson"),
       pch = c(22, 16),
       pt.bg = c("grey", NA), 
       col = c("black", "red"), 
       bty = "n")

## Part D: T_4 empirical pdf vs. true pdf
sim_T4 <- rgamma(M, shape = 4, rate = lambda)

## empirical density estimate
hist(sim_T4,
     breaks = 60, probability = TRUE,
     xlab = expression(t),
     main = expression("Jump time  T"[4]*": empirical pdf vs. true pdf"))
# overlay true Gamma density
curve(dgamma(x, shape = 4, rate = lambda),
      from = 0, to = max(sim_T4),
      add = TRUE, lwd = 2, n = 1000, col = "red")
legend("topright", legend = c("Histogram", "True Γ(4,λ)"),
       fill = c("grey", NA), border = c("black", NA), lwd = c(NA, 2),
       col = c("black", "red"), bty = "n")
