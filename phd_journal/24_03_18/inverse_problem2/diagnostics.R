table <- read.csv("/Users/saraiva/PycharmProjects/IT-LongTermBiosignals/phd_journal/24_03_18/inverse_problem/predictions.csv", header = TRUE, sep = ",")
y <- table$y
yhat <- table$yhat

# Linear model
rhlm <- lm(y ~ yhat, table)
summary(rhlm)


