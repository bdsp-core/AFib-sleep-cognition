library(reticulate)
library(EValue)

folder <- 'D:\\projects\\AFib-sleep-cognition\\MrOS-analysis'
suffix <- '_notholdAHI2'

data.path <- file.path(folder, sprintf('AF_as_exposure_potential_outcomes%s.pickle', suffix))
dat <- py_load_object(data.path)

feature.path <- file.path(folder, 'figures', sprintf('significant_features%s.csv', suffix))
df.feat <- read.csv(feature.path)
df <- data.frame()
for (i in 1:nrow(df.feat)) {
  feature.name <- df.feat$Name[i]
  method <- df.feat$method[i]
  key <- sprintf("('%s', '%s')", feature.name, method)
  x0 <- as.vector(dat[[key]][[1]])
  x1 <- as.vector(dat[[key]][[2]])
  
  cd <- cohensD(x0,x1)
  se <- sd(x1-x0)/sqrt(length(x1))
  ev <- evalues.MD(cd, se)
  
  df[i,'Name'] <- feature.name
  df[i,'method'] <- method
  df[i,'EValPoint'] <- ev['E-values','point']
  df[i,'EvalLB'] <- ev['E-values','lower']
}
print(df)
result.path <- file.path(folder, 'tables', sprintf('sensitivity_result%s.csv', suffix))
write.csv(df, result.path, row.names=F)
