from keras.models import load_model
import shutil

# Define the directory you want to zip
directory_to_zip = '/kaggle/working/'

# Define the name for the zip file
zip_file_name = '/kaggle/working/FER_CNN_LeNet-5'

# Create a zip file of the directory
shutil.make_archive(zip_file_name, 'zip', directory_to_zip)

# Alternatively, if you want to save the model directly in the zip file:
# Save the model
model.save('/kaggle/working/FER_CNN_LeNet-5.h5')
