# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, 
                    epochs = 30, 
                    batch_size = 48, 
                    validation_data = (test_x, test_y), 
                    verbose=1)
