import numpy as np
from PIL import Image
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report

#Loading the images into numpy arrays

train_images = []
for i in range (1, 15001):
    png = f"data/data/{i:06d}.png"
    image = np.array(Image.open(png).convert('L')) / 255
    train_images.append(image)
np.save("data/numpy_data/train_images.npy", train_images)


val_images = []
for i in range (15001, 17001):
    png = f"data/data/{i:06d}.png"
    image = np.array(Image.open(png).convert('L')) / 255
    val_images.append(image)
np.save("data/numpy_data/val_images.npy", val_images)


test_images = []
for i in range (17001, 22150):
    png = f"data/data/{i:06d}.png"
    image = np.array(Image.open(png).convert('L')) / 255
    test_images.append(image)
np.save("data/numpy_data/test_images.npy", test_images)


#Loading the labels and creating a numpy file with the informations

labels_train = []
skips = False
with open ("train_labels.txt", "r") as file:
    for line in file:
        if not skips:
            skips = True
            continue
        labels_train.append(int(line.split(",")[1]))

np.save("data/numpy_data/train_labels.npy", labels_train)


labels_val = []
skips = False
with open ("validation_labels.txt", "r") as file:
    for line in file:
        if not skips:
            skips = True
            continue
        labels_val.append(int(line.split(",")[1]))

np.save("data/numpy_data/val_labels.npy", labels_val)

train_images = np.load("data/numpy_data/train_images.npy")
train_images = train_images.reshape(train_images.shape[0], -1)

val_images = np.load("data/numpy_data/val_images.npy")
val_images = val_images.reshape(val_images.shape[0], -1)

labels_train = np.load("data/numpy_data/train_labels.npy")
labels_val = np.load("data/numpy_data/val_labels.npy")

#Training the Naive Bayes model

clf = MultinomialNB()
clf.fit(train_images,labels_train)

print("F1 score : ", f1_score(labels_val, clf.predict(val_images)))

#Printing The CSV file (the labeling that the model did)

with open("Submission.csv", "w") as file:
    file.write("id,class\n")
    for i in range (17001, 22150):
        file.write(f"0{i},{clf.predict(test_images[i-17001].reshape(1, -1))[0]}\n")

# Generate predictions on the validation set
val_pred = clf.predict(val_images)

# Generate confusion matrix
cm = confusion_matrix(labels_val, val_pred)

# Generate classification report
report = classification_report(labels_val, val_pred, target_names=["tumor", "non_tumor"])

# Write confusion matrix to file
with open("confusion_matrix.csv", "w") as file:
    file.write("actual_tumor,actual_non_tumor\n")
    file.write(f"predicted_tumor,{cm[0,0]},{cm[0,1]}\n")
    file.write(f"predicted_non_tumor,{cm[1,0]},{cm[1,1]}\n\n\n")
    file.write(report)

