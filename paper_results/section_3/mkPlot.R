library(ggplot2)

X1 <- read.delim("volume_estimation_by_batch_size_MNIST.csv", sep = ",")
X2 <- read.delim("volume_estimation_by_number_of_layers_MNIST.csv", sep = ",")
X3 <- read.delim("volume_estimation_by_number_of_neurons_MNIST.csv", sep = ",")
X4 <- read.delim("volume_estimation_by_sample_size_MNIST.csv", sep = ",")


sSize <- c(1000, 60000)

tempsSize <- 10^seq(from = log(sSize[1]) / log(10), to = log(sSize[2]) / log(10), length.out = 8)

tempsSize <- round(tempsSize / 1000) * 1000
tempsSize

sSize <- c(1000,10000,20000,30000,40000,50000, 60000)


X4pre <- data.frame(sample_size = c(sSize,sSize,sSize))
X4pre$precision <- as.numeric(t(matrix(c(16,32,64), 3, length(sSize))))
X4pre$X <- 0:(nrow(X4pre)-1)
X4pre$count <- 0
X4pre <- X4pre[,c(3,2,4,1)]


batch_size = 256
for(k in 1:nrow(X4pre)){
	Xtemp <- X4
	Xtemp <- Xtemp[X4$precision == X4pre$precision[k],]
	X4pre$count[k] <- 1 - sum(Xtemp$batch_idx == floor( X4pre$sample_size[k] / batch_size) ) / 1000
}

X4pre$count[X4pre$precision == 64] <- 0
X4pre$sample_size <- X4pre$sample_size /1000
X4 <- X4pre


X1 <- rbind(X1, X1[1:5,])
X1$precision[11:15] <- 64
X1$count[11:15] <- 0
X1$batch_size[11:15] <- unique(X1$batch_size)

X1$count <- X1$count / max(X1$count) * 100
X2$count <- X2$count / max(X2$count) * 100
X3$count <- X3$count / max(X3$count) * 100
X4$count <- X4$count / max(X4$count) * 100



X1$type = as.factor("Batch size")
X2$type = as.factor("Number of layers")
X3$type = as.factor("Number of neurons")
X4$type = as.factor("Sample size (1000s)")


## Remove spurious columns
X3 <- X3[,-4]
X1 <- X1[,-2]


xVar <- "xVar"
names(X1)[4] <- xVar
names(X2)[4] <- xVar
names(X3)[4] <- xVar
names(X4)[4] <- xVar

yVar <- "Proportion"
names(X1)[3] <- yVar
names(X2)[3] <- yVar
names(X3)[3] <- yVar
names(X4)[3] <- yVar



toPlot <- rbind(X4,X3,X2,X1)
toPlot$Proportion[toPlot$precision == 64] <- 0.5



pl <- ggplot(data = toPlot) + 
			geom_bar(aes(x = as.factor(xVar), y = Proportion, fill = as.factor(precision)),stat = "identity", position = "dodge") + 
			facet_wrap(~type, ncol = 4, scales = "free_x", strip.position = "bottom") + 
			theme_bw() + 
			xlab(label = "") + 
			ylab(label = "Proportion %") + 
			scale_fill_manual(name = "Precision", values = c("#3274a1", "#e1812c", "#3a923a")) +  
			scale_y_continuous(breaks=(0:5)*20)

pdf("volumeMNIST.pdf", width = 8, height = 2)
print(pl)
dev.off()



X <- read.delim("norm_diff_estimation_MNIST.csv", sep = ",")


toPlot <- X
toPlot <- toPlot[toPlot$precision == 32,]
toPlot2 <- toPlot
toPlot2 <- toPlot2[toPlot2$same == "False",]


nAffect <- 0

for (run in unique(toPlot$run_id)){
	dataTemp <- toPlot[toPlot$run_id == run,] 
	nAffect <- nAffect + (sum(dataTemp$diff_L1>0)>0) * 100
}

print(paste("Precision 32: ", as.character(nAffect/max(toPlot$run_id + 1) ), "% run affected"))

nAffect <- sum(toPlot$diff_L1>0) / nrow(toPlot) * 100
print(paste("Precision 32: ", as.character(nAffect/max(toPlot$run_id + 1) ), "% minibatch affected"))

print("Precision 32, relative difference:")
print(summary(toPlot2$diff_L2 / toPlot2$net1_L2))



toPlot <- X
toPlot <- toPlot[toPlot$precision == 16,]
toPlot2 <- toPlot
toPlot2 <- toPlot2[toPlot2$same == "False",]


nAffect <- 0

for (run in unique(toPlot$run_id)){
	dataTemp <- toPlot[toPlot$run_id == run,] 
	nAffect <- nAffect + (sum(dataTemp$diff_L1>0)>0) * 100
}

print(paste("Precision 16: ", as.character(nAffect/max(toPlot$run_id + 1) ), "% run affected"))

nAffect <- sum(toPlot$diff_L1>0) / nrow(toPlot) * 100
print(paste("Precision 16: ", as.character(nAffect/max(toPlot$run_id + 1) ), "% minibatch affected"))

print("Precision 16, relative difference:")
print(summary(toPlot2$diff_L2 / toPlot2$net1_L2))


