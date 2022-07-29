library(RandomFields)
library(reticulate)
use_python("~/software/anaconda3/envs/pytorch/bin/python3")

args = commandArgs(trailingOnly=TRUE)

outPath <- args[1]
n <- as.integer(args[2])
dimX <- as.integer(args[3])
dimY  <- as.integer(args[4])
rfType <- args[5]
covType  <- args[6]
df <- as.integer(args[7])
scale <- as.numeric(args[8])
thresh <- as.numeric(args[9])
outPathMask <- args[10]
prop <- as.numeric(args[11])

RFoptions(seed=NA) #seed=NA to make them random
covar_fun <- RMgauss() #RMgauss()=e^{−r^2} RMexp()=e^{−r}
model <- RPgauss(RMgauss(scale=scale))
x <- seq(1, dimX, 1)
y <- seq(1, dimY, 1)


print(thresh)
print(prop)
##for(i in 1:n){
i <- 0
counter <- 0
pval <- 0
while(i<n){
    print(paste0("RF: ", i))
    if (rfType == "chisq"){
        z <- RFsimulate(model, x, y, n=df)
        rf <- matrix(rowSums(z@data**2), dimX, dimY)
        #rf <- rf/sum(rf)
    } else {
        z <- RFsimulate(model, x, y, n=1)
        ## rf <-  matrix(rowSums(z@data), dimX, dimY)yes
        rf <-  matrix(z@data[, 1], dimX, dimY)
    }
   
    ## i <- i + 1
    ## if (max(rf)>thresh) {counter <- counter +1}
    ## py_save_object(array(rf, dim = c(1, 1, dimX, dimY)), paste0(outPath, i, ".pickle"))


    if (max(rf) >= thresh){pval <- pval+1/n}
    
    if (counter < n*prop){
        if (max(rf) < thresh){next} else {counter <- counter + 1; i <- i + 1}
    } else {i <- i + 1}
    py_save_object(array(rf, dim = c(1, 1, dimX, dimY)), paste0(outPath, i, ".pickle"))
    
    #Generate mask
    mask <- 0*rf
    mask[rf >= thresh] <- 1
    py_save_object(array(mask, dim = c(1, 1, dimX, dimY)), paste0(outPathMask, i, ".pickle"))
    
}
print(paste0(outPath, ": P-Value: ", pval))
