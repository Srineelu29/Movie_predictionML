library(keras)

FLAGS= flags(
  flag_numeric("learning_rate", 0.01),
  flag_numeric("units1", 32),
  flag_numeric('units2', 64),
  flag_numeric("units3", 128),
  flag_numeric("batch_size", 32),
  flag_numeric("dropout", 0.3)
)

model_mv <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$units1, activation = "relu", input_shape = ncol(train_data)-1) %>%
  layer_dropout(FLAGS$dropout) %>%
  layer_dense(units = FLAGS$units2, activation = "relu") %>%
  layer_dropout(FLAGS$dropout) %>%
  layer_dense(units = 1, activation = "sigmoid")

opt= optimizer_sgd(learning_rate= FLAGS$learning_rate)
model_mv %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam",
  metrics = "accuracy",
)

histor_mv <- model_mv %>% fit(as.matrix(train_data[,1:6]), 
                             as.matrix(train_data[,7]), 
                             epochs = 100, 
                             batch_size = 100,
                             validation_data = list(as.matrix(val_data[,1:6]), as.matrix(val_data[,7])),
                             verbose = 1)
                             
                             