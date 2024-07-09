

# Read the CSV file
data <- read.csv("/Users/saraiva/PycharmProjects/LTBio/phd_journal/24_03_18/inverse_problem3/scheme56/predictions_targets.csv")

# Fit a linear model
model <- lm(predictions ~ targets, data = data)
summary(model)

# Plot residuals vs. leverage
plot(model, which = 5)

qqnorm(resid(model)) # A quantile normal plot - good for checking normality
qqline(resid(model))
