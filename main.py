import numpy as np
import os
import sys
from tempCNN_model import TempCNN_Model
import tensorflow as tf
import time
from sklearn.utils import shuffle
os.environ['AUTOGRAPH_VERBOSITY'] = '1'

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def buildPair(x_train, labels):
    f_data = []
    s_data = []
    y_val = []
    n_examples = labels.shape[0]
    f_y = []
    s_y = []
    for i in range(n_examples):
        for j in range(i+1, n_examples):
            if labels[i] == labels[j]:
                y_val.append(0)
            else:
                y_val.append(1)
            f_data.append( x_train[i])
            s_data.append( x_train[j])
            f_y.append( labels[i])
            s_y.append(labels[j])
    return np.stack(f_data, axis=0), np.stack(s_data, axis=0), np.array(y_val), np.array(f_y), np.array(s_y)

def getBatch(X, i, batch_size):
	start_id = i*batch_size
	t = (i+1) * batch_size
	end_id = min( (i+1) * batch_size, X.shape[0])
	batch_x = X[start_id:end_id]
	return batch_x

def SampleNegPairs(pred, data, current_b_size):
    beta = .9
    max_prob = np.amax(pred,axis=1)
    idx = np.where(max_prob > beta)[0]
    pl = np.argmax(pred,axis=1)
    sub_pl = pl[idx]
    sub_data = data[idx]
    f_data = []
    s_data = []
    y_val = []
    length = len(sub_pl)
    for i in range(length-1):
        for j in range(i,length):
            f_data.append(sub_data[i])
            s_data.append(sub_data[j])
            if sub_pl[i] != sub_pl[j]:
                y_val.append(1)
            else:
                y_val.append(0)
                
    f_data = np.array(f_data)
    s_data = np.array(s_data)
    y_val = np.array(y_val)
    f_data, s_data, y_val = shuffle(f_data, s_data, y_val)
    return f_data[0:current_b_size], s_data[0:current_b_size], y_val[0:current_b_size]

def train3(unl_data, f_data, s_data, binary_val, f_y, s_y, model, opt, cl_loss, mae_loss, BATCH_SIZE, e):
    tot_loss = 0
    margin = 1.0
    iterations = f_data.shape[0] / BATCH_SIZE
    if f_data.shape[0] % BATCH_SIZE != 0:
        iterations += 1

    for ibatch in range(int(iterations)):
        batch_f = getBatch(f_data, ibatch, BATCH_SIZE)
        batch_s = getBatch(s_data, ibatch, BATCH_SIZE)
        batch_unl = getBatch(unl_data, ibatch, BATCH_SIZE)
        flatten_batch_s = np.reshape(batch_s,(batch_s.shape[0],-1))
        flatten_batch_f = np.reshape(batch_f,(batch_f.shape[0],-1))
        flatten_batch_unl = np.reshape(batch_unl,(batch_unl.shape[0],-1))
           
        batch_f_y = getBatch(f_y, ibatch, BATCH_SIZE)
        batch_s_y = getBatch(s_y, ibatch, BATCH_SIZE)
        batch_y = getBatch(binary_val, ibatch, BATCH_SIZE)
        current_b_size = batch_f_y.shape[0]
        alpha = 10
        with tf.GradientTape() as tape:
            d_w = model.siameseDistance([batch_f, batch_s], training=True)
            equal_loss = (.5* (1-batch_y) * d_w)
            neg_loss = (.5* batch_y * tf.math.maximum(0 , margin - d_w) )
            loss_metric = alpha * tf.math.reduce_mean( equal_loss + neg_loss )

            pred_f, reco_f, _ = model(batch_f, training=True)
            pred_s, reco_s, _ = model(batch_s, training=True)
            pred_unl, reco_unl, _ = model(batch_unl, training=True)

            loss_reco = mae_loss(flatten_batch_f, reco_f) + mae_loss(flatten_batch_s, reco_s)
            
            loss_cl = cl_loss(batch_f_y, pred_f) + cl_loss(batch_s_y, pred_s)

            loss_labelled = loss_metric + loss_reco + loss_cl
            loss = loss_labelled
            
            if e > 50:
                loss_unl = mae_loss(flatten_batch_unl, reco_unl)
                batch_f_unl, batch_s_unl, batch_y_unl = SampleNegPairs(pred_unl.numpy(), batch_unl, current_b_size)
                d_w_unl = model.siameseDistance([batch_f_unl, batch_s_unl], training=True)
                equal_loss_unl = (.5* (1-batch_y_unl) * d_w_unl)
                neg_loss_unl = (.5* batch_y_unl * tf.math.maximum(0 , margin - d_w_unl) )
                loss_metric_unl = alpha * tf.math.reduce_mean( equal_loss_unl + neg_loss_unl )
                loss_unl += loss_metric_unl
                loss += loss_unl
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]
            opt.apply_gradients(zip(grads, model.trainable_variables))
            
            tot_loss+=loss
    return (tot_loss / iterations)

def main(argv):
    #Directory in which data are stored
    dataDir = argv[1]
    #number of labelled samples to access data information
    nSamples = argv[2]
    #run identifier to add to the output file name
    runId = argv[3]

    n_epochs = 300
    path = dataDir+"/OUR"
    path_model = dataDir+"/OUR_MODEL"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path_model)

    data = np.load(dataDir+"/data.npy")
    labels = np.load(dataDir+"/labels.npy")
    #labels = labels[:,2]-1
    
    n_classes = len(np.unique(labels))
    idxLabelledData = np.load(dataDir+"/labels_%s_%s.npy"%(runId, nSamples),allow_pickle=True)
    idxLabelledData = idxLabelledData[:,0]

    labelledData = data[idxLabelledData]
    labelsSelected = labels[idxLabelledData]

    first_data, second_data, binary_val, f_y, s_y = buildPair(labelledData, labelsSelected)

    emb_size = 1344

    cPerClasses = emb_size // n_classes
    protos = np.zeros((n_classes,emb_size))
    for i in range(n_classes):
        begin_i = i * cPerClasses 
        end_i = (i+1) * cPerClasses
        protos[i,begin_i:end_i] = 1.0

    nUnit2Reco = (labelledData.shape[1] * labelledData.shape[2])
    tempCNN = TempCNN_Model(n_classes, nUnit2Reco, flatten=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    cl_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    
    to_retain = np.arange(data.shape[0])
    to_retain = np.setdiff1d(to_retain, idxLabelledData)    

    tempData = np.array(data)

    start_global = time.time()
    for e in range(n_epochs):
        loss2 = 0
        loss = 0
        start = time.time()
        labelledData, labelsSelected = shuffle(labelledData, labelsSelected)
        tempData = shuffle(tempData)

        first_data, second_data, binary_val, f_y, s_y = shuffle(first_data, second_data, binary_val, f_y, s_y)        
        temp_first_data = first_data[0:tempData.shape[0]]
        temp_second_data = second_data[0:tempData.shape[0]]
        temp_binary_val = binary_val[0:tempData.shape[0]]
        temp_f_y = f_y[0:tempData.shape[0]]
        temp_s_y = s_y[0:tempData.shape[0]]
        loss2 = train3(tempData, temp_first_data, temp_second_data, temp_binary_val, temp_f_y, temp_s_y, tempCNN, optimizer, cl_loss, mae_loss, 256, e)

        end = time.time()
        print("epoch %d with loss %.4f loss2 %.4f in %d seconds"%(e, loss, loss2, (end-start)))  

    end_global = time.time()
    elapsed = end_global - start_global
    np.save(path+"/time_%s_%s.npy"%(runId, nSamples), elapsed)

    tempCNN.save_weights(path_model+"/model_%s_%s"%(runId,nSamples))

    pred, _, _ = tempCNN.predict(data, batch_size=2048)
    pred = np.argmax(pred,axis=1)
    np.save(path+"/pred_%s_%s.npy"%(runId,nSamples),pred)


if __name__ == "__main__":
   main(sys.argv)