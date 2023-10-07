import os.path

from __config__ import *
from __imports__ import *


print("Loading Model!")
model = load_model(os.path.join(model_output_path, model_name))
print("Model Loaded!")

#Normalizing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# loading testing data
testdf = pd.read_csv(test_csv_path)
test_set = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=input_data_path+'/TestSet',
        x_col="id",
        y_col="labels",
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        class_mode=class_mode,
        target_size=target_size)


# Confusion Matrix and Classification report
Y_pred = model.predict(test_set, test_set.samples//32 +1)
y_pred = np.argmax(Y_pred, axis=1)

# encode model predictions
testdf['prediction'] = ['positive' if x == 1 else 'negative' for x in y_pred]

# save model predictions to an output file
testdf.to_csv(os.path.join(model_output_path, 'model_predictions.csv'), index=False)

# confusion matrix and classification reports

target_names = ['negative', 'positive']
classifcn_report = classification_report(test_set.classes, y_pred, target_names=target_names)
print(classifcn_report)


cm = confusion_matrix(test_set.classes, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot()
plt.savefig(os.path.join(model_output_path, 'test_confusion_matrix.png'))

