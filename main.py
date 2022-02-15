from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "226d3a90-8e7e-11ec-bb9e-1bfe694696054d3cb640-e887-4f7f-9945-e445a497a72d"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("my-test-image.jpg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))

