simulate_poisson_times <- function(lambda, T) {
  times <- c()
  t <- 0
  while (TRUE) {
    t <- t + rexp(1, rate = lambda)  
    if (t > T) break
    times <- c(times, t)
  }
  return(times)
}

poisson_path <- function(lambda, T, dt = 0.01) {
  arrivals <- simulate_poisson_times(lambda, T)
  grid <- seq(0, T, by = dt)
  counts <- findInterval(grid, arrivals)
  data.frame(t = grid, Nt = counts)
}

set.seed(123)   
lambda <- 3
T <- 10

path_df <- poisson_path(lambda, T)

plot(path_df$t, path_df$Nt,
     type = "s",
     main = bquote("Cammino Poisson: " ~ lambda == .(lambda) ~ ",  T == " ~ .(T)),
     xlab = "t",
     ylab = expression(N[t]),
     lwd = 2)

