import numpy as np
from PIL import Image
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

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

#Training the Nayve Bayes model

clf = MultinomialNB()
clf.fit(train_images,labels_train)

print("F1 score : ", f1_score(labels_val, clf.predict(val_images)))

#Printing The CSV file (the labeling that the model did)

with open("Submission.csv", "w") as file:
    file.write("id,class\n")
    for i in range (17001, 22150):
        file.write(f"0{i},{clf.predict(test_images[i-17001].reshape(1, -1))[0]}\n")

#Creating numpy files for the tumor images and non-tumor images so that
#we can train the random forest model on a more uniform labeling

tumor_images = []
non_tumor_images = []
for i in range (1, 15001):
    if (labels_train[i-1] == 1):
        png = f"data/data/{i:06d}.png"
        image = np.array(Image.open(png).convert('L')) / 255
        tumor_images.append(image)
    else:
        png = f"data/data/{i:06d}.png"
        image = np.array(Image.open(png).convert('L')) / 255
        non_tumor_images.append(image)

np.save("data/numpy_data/tumor_images.npy", tumor_images)
np.save("data/numpy_data/non_tumor_images.npy", non_tumor_images)

tumor_images = np.load("data/numpy_data/tumor_images.npy")
tumor_images = tumor_images.reshape(tumor_images.shape[0], -1)

non_tumor_images = np.load("data/numpy_data/non_tumor_images.npy")
non_tumor_images = non_tumor_images.reshape(non_tumor_images.shape[0], -1)

print (len(tumor_images))
print (len(non_tumor_images))


import random

#Concate the tumor and non-tumor images and randomizing the order of the images and labels in the same way

mixed_images = np.concatenate((tumor_images, non_tumor_images[0:4800]))
mixed_labels = np.concatenate((np.ones(len(tumor_images)), np.zeros(len(non_tumor_images[0:4800]))))

print (len(mixed_images))
print (len(mixed_labels))


idx = np.arange(len(mixed_labels))
np.random.shuffle(idx)
mixed_images = mixed_images[idx]
mixed_labels = mixed_labels[idx]


print (len(mixed_images))

# The improved random forest clasifier 

clf_forest = RandomForestClassifier()
clf_forest.fit(mixed_images, mixed_labels)

# Evaluate the classifier on the validation set
val_pred = clf_forest.predict(val_images)
print("F1 score : ", f1_score(labels_val, val_pred))


# # Save the test predictions to a file

with open("Random_forest_4802.csv", "w") as file:
    file.write("id,class\n")
    for i in range(17001, 22150):
        label = clf_forest.predict(test_images[i - 17001].reshape(1, -1).astype(int))[0]
        file.write(f"{i},{int(label)}\n")

# Generate predictions on the validation set
val_pred = clf_forest.predict(val_images)

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


# with open("Random_forest_4802.csv", "w") as file:
#     file.write("id,class\n")
#     for i in range (17001, 22150):
#         file.write(f"0{i},{clf_forest.predict(test_images[i-17001].reshape(1, -1).astype(int))[0]}\n")



#The basic random forest classifier

# clf_forest = RandomForestClassifier()
# clf_forest.fit(train_images,labels_train)

# print("F1 score : ", f1_score(labels_val, clf_forest.predict(val_images)))

# with open("Random_forest_25.csv", "w") as file:
#     file.write("id,class\n")
#     for i in range (17001, 22150):
#         file.write(f"0{i},{clf_forest.predict(test_images[i-17001].reshape(1, -1))[0]}\n")