library(matrixcalc)
# Transition matrix
transMat <- matrix(c(0,0,1,0,0,
                     1,0,0,0,0,
                     0,.5,0,.5,0,
                     0,0,.5,0,.5,
                     0,0,0,1,0),
                   byrow = TRUE, ncol = 5)

MC5stateProbability <- function(m, j, i, transMat){
  Pm <- matrix.power(transMat, m)   # matrix.power
  Pm[i, j]                  
}

MC5stateProbability(4, 3, 3, transMat)

l<-10
matrix.power(transMat,l)
l<-100
matrix.power(transMat,l)
l<-1000
matrix.power(transMat,l)