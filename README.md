# S3ITS
Code associated to the paper https://doi.org/10.1016/j.neucom.2023.127031

There are two files:
- **tempCNN_model.py**
- **main.py**

The first python file contains the definition of the backbone (TempCNN) architecture while the second file contains the training procedure for the Constrastive Semi-Supervised Deep Learning framework.

To run the main file, here an example:
  python main.py koumbia 10 0

Where:
  - **koumbia** is the directory where the data, labels and the id set of training examples is contained
  - **10** is the number of per class labelled samples to use to train the model
  - **0** is the runID

The scritp will looking for the following fils in the koumbia directory:
- **data.npy** file that contains the data with a shape: #Samples X #Timestamps X #Bands
- **labels.npy** file that contains the labels associated to each samples, with a shape: #Samples
- **labels_0_10.npy** file that contains the ids (indexes) of the samples to use as training with the corresponding class value (#Classes x 10) X 2. The first column is the row id w.r.t. the data.npy file and the second column is the class value.


The method is implemented with Tensorflow 2.
