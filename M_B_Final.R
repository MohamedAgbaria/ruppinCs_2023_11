library(bnlearn)
if (!require("BiocManager", quietly = TRUE))
  +     install.packages("BiocManager")
mydata <- read.csv("C:/Users/win10/Desktop/תשפג סמסטר א/פרויקט גמר חלק א'/חלק ב/R_Final_Project/MB_Final/data.csv")
bn = mmhc(mydata)
c = bn
graphviz.plot(c)
bn_mmhc = bn.fit(c,mydata)
mb_ = mb(bn_mmhc , "target")
mb_df = mydata[,c(mb_,"target")]
c2 = mmhc(mb_df)
graphviz.plot(c2)
bn_mmhc_2 = bn.fit(c2,mb_df)
plot(c2, main = "Markov Blanket", highlight = c("target"))
plot(c, main = "all", highlight = c("target"))

