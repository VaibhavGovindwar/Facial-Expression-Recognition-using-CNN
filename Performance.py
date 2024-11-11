from sklearn.metrics import accuracy_score

# evaluate the test performance

# Convert one-hot encoded test labels back to categorical labels
test_true = np.argmax(test_y,axis=1)
# Predict the labels for the test data
test_pred = np.argmax(model.predict(test_x), axis = 1)
# Compute the accuracy score for the test data
# Print the test accuracy
print("Test Accuracy {:.2f}".format(accuracy_score(test_true, test_pred)*100))

# Evaluate the train performance

# Convert one-hot encoded train labels back to categorical labels
train_true = np.argmax(train_y,axis=1)
# Predict the labels for the train data
train_pred = np.argmax(model.predict(train_x), axis = 1)
# Compute the accuracy score for the train data
# Print the train accuracy
print("Train Accuracy {:.2f}".format(accuracy_score(train_true, train_pred)*100))
