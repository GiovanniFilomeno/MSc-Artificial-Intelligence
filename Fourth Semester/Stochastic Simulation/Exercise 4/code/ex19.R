## Parameters
T_tot <- 30           # time horizon for the path in part (b)
M     <- 1e4          # Monte-Carlo replications for part (c)

lambda_t <- function(t) 5 * sin(t / 2)^2     
Lambda_t <- function(t) 2.5 * (t - sin(t))         

## Part A: generating path
simulate_NHPP_path <- function(T_end,
                               lambda_fun  = lambda_t,
                               lambda_max  = 5)
{
  t  <- 0           # current time
  N  <- 0           # current count
  ts <- c(0)        # jump times (incl. origin)
  Ns <- c(0)        # counts    (incl. N₀ = 0)
  
  while (TRUE) {
    w <- rexp(1, rate = lambda_max)
    t <- t + w
    if (t > T_end) break
    
    ## thinning step
    if (runif(1) < lambda_fun(t) / lambda_max) {
      N  <- N + 1
      ts <- c(ts, t)
      Ns <- c(Ns, N)
    }
  }
  
  ts <- c(ts, T_end)
  Ns <- c(Ns, N)
  
  data.frame(time = ts, count = Ns)
}

## Part B: draw one path (T = 30)
set.seed(1)
path <- simulate_NHPP_path(T_tot)

plot(path$time, path$count, type = "s",
     main = bquote("NHPP path,  λ(t)=5·sin²(t/2),   T="*.(T_tot)),
     xlab = "time t", ylab = expression(N[t]))
grid()

## Part C: empirical vs. true pmf of  Ns
T_check <- 5

### Simulate N₅ M=10^4 times using previous algorithm
sim_N5 <- replicate(
  M,
  {
    df <- simulate_NHPP_path(T_check)
    tail(df$count, 1)           # last count is N₅
  }
)

### Empirical pmf
pmf_hat <- table(sim_N5) / M

barplot(pmf_hat,
        names.arg = names(pmf_hat),
        xlab = expression(k), ylab = "Probability",
        main = expression("Empirical pmf of  N"[5]))

### Comparison
lambda_bar <- Lambda_t(T_check)           # mean of the Poisson law
k_vals <- as.integer(names(pmf_hat))

points(1:length(k_vals),
       dpois(k_vals, lambda_bar),
       pch = 16, col = "red")
legend("topright",
       legend = c("Empirical", paste0("Poisson(Λ(5)) , Λ(5)≈", round(lambda_bar,3))),
       pch = c(22, 16), pt.bg = c("grey", NA), col = c("black", "red"), bty = "n")

