FLAGS= flags(
  flag_numeric("learning_rate", 0.01),
  flag_numeric("units1", 256),
  flag_numeric('units2', 128),
  flag_numeric("batch_size", 32),
  flag_numeric("epochs", 20)
)

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$units1, activation = "relu",
              input_shape = dim(train_X)[2]) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = FLAGS$units2, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1)

opt= optimizer_adam(learning_rate=FLAGS$learning_rate)
model %>% compile(
  loss = "mse",
  optimizer = opt,
  metrics= "mse")

history <- model %>% fit(as.matrix(train_X),
                         train_y,
                         batch_size=FLAGS$batch_size,
                         epochs = FLAGS$epochs, 
                         verbose=2,
                         validation_data=list(as.matrix(valid_X), valid_y)
)