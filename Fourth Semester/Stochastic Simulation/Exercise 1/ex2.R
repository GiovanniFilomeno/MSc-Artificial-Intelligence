setwd("/Users/Q540900/Desktop/A.I. Master/Fourth Semester/Stochastic Simulation/Exercise 1")
lcg <- function(a, c, m, x0) {
  vals <- integer(0)  # storing the generated value
  curr <- x0 # Initialization         
  
  repeat {
    next_val <- (a * curr + c) %% m # Generating next value
    vals <- c(vals, next_val) # Store the generated vyalue
    curr <- next_val # Update the state
    
    if (curr == x0) { # Stop in case original value found
      break
    }
  }
  
  return(vals)
}

# Exercise 2.b
seq_lcg <- lcg(a = 7, c = 0, m = 31, x0 = 19)
# Printing the sequence
seq_lcg

# Exercise 3.a
seq_a <- lcg(a = 17, c = 0, m = 2^13 - 1, x0 = 1)
seq_b <- lcg(a = 29, c = 0, m = 2^13 - 1, x0 = 1)
seq_c <- lcg(a = 197, c = 0, m = 2^13 - 1, x0 = 1)

# Check lengths
cat("Lengths:\n",
    "  a=17 ->", length(seq_a), "\n",
    "  a=29 ->", length(seq_b), "\n",
    "  a=197->", length(seq_c), "\n")

if (!dir.exists("images")) {
  dir.create("images")
}

# 4) Plot (X_n, X_{n+1}) for each sequence and save to .png

# --- a=17 ---
df_a <- data.frame(x = seq_a[-length(seq_a)],
                   y = seq_a[-1])
png("plots/a_17.png", width=800, height=600, res=96)
plot(df_a$x, df_a$y, pch=16, cex=0.5,
     main="Pairs (X_n, X_{n+1}) for a=17",
     xlab="X_n", ylab="X_{n+1}")
dev.off()

# --- a=29 ---
df_b <- data.frame(x = seq_b[-length(seq_b)],
                   y = seq_b[-1])
png("plots/a_29.png", width=800, height=600, res=96)
plot(df_b$x, df_b$y, pch=16, cex=0.5,
     main="Pairs (X_n, X_{n+1}) for a=29",
     xlab="X_n", ylab="X_{n+1}")
dev.off()

# --- a=197 ---
df_c <- data.frame(x = seq_c[-length(seq_c)],
                   y = seq_c[-1])
png("plots/a_197.png", width=800, height=600, res=96)
plot(df_c$x, df_c$y, pch=16, cex=0.5,
     main="Pairs (X_n, X_{n+1}) for a=197",
     xlab="X_n", ylab="X_{n+1}")
dev.off()

cat("Plots saved to 'plots/' directory.\n")

acf(seq_a, main="ACF of sequence (a=17)")
acf(seq_b, main="ACF of sequence (a=29)")
acf(seq_c, main="ACF of sequence (a=197)")

png("plots/acf_a17.png", width = 800, height = 600, res = 96)

acf(seq_a, main="ACF of sequence (a=17)")

dev.off()

png("plots/acf_a29.png", width = 800, height = 600, res = 96)

acf(seq_b, main="ACF of sequence (a=29)")

dev.off()

png("plots/acf_a197.png", width = 800, height = 600, res = 96)

acf(seq_c, main="ACF of sequence (a=197)")

dev.off()

