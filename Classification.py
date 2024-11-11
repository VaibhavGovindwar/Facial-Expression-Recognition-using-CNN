from sklearn.metrics import classification_report
from tabulate import tabulate

report = classification_report(test_true, test_pred, target_names=emotion_labels)
title='Classification Report'
# Center the title
title_length = len(title)
header = f"{'':^{title_length}}\n{title:^{title_length}}\n"

print(header)
print(report)
