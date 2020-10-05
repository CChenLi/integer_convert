data = read.csv('data/chinese_mnist.csv')

create_path <- function(x) {
  suit <- as.character(x["suite_id"])
  sam <- as.character(x["sample_id"])
  code <- as.character(x["code"])
  res <- paste("data/images/input_", suit, "_", sam, "_", code, ".jpg", sep="")
  sentenceString = gsub(" ","",res)
  return(sentenceString)
}

path = apply(data, 1, create_path)

new_data = cbind(path, data$value)
set.seed(42)
perm <- sample(nrow(new_data))
new_data <- new_data[perm, ]

write.csv(new_data, "data/processed.csv")
