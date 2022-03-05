'''
Created on 26 nov 2021

@author: main
'''

import numpy as np
import os

def get_label(fn):
    fn_ext = os.path.basename(fn)
    return os.path.splitext(fn_ext)[0]
    

def fix_dims(f1):
    if f1.ndim == 1:
        F1 = np.expand_dims(f1, axis = 0)
    elif f1.ndim == 2:
        F1 = f1
    else:
        raise Exception("features dims should be 1 or 2")
    
    return F1    

def COS(f1, f2, _ord_ = 2) :
    
    F1 = fix_dims(f1)
    F2 = fix_dims(f2)  
    
    if not _ord_ is None:
        qf_norms = np.linalg.norm(F1, ord = _ord_, axis=1)
        qf = F1 / qf_norms[:, np.newaxis]
        
        rf_norms = np.linalg.norm(F2, ord = _ord_, axis=1)
        rf = F2 / rf_norms[:, np.newaxis]
    else:
        qf = F1
        rf = F2
    
    qf = np.reshape(qf, newshape = (np.shape(qf)[0], -1) , order = 'C')
    rf = np.reshape(rf, newshape = (np.shape(rf)[0], -1) , order = 'C')
    
    # axis = 0 contains QUERY
    # axis = 1 contains the distances from a query and each of the references
    # The element 5,15 is the distance between the 5th query image and the 15th reference
    cos = np.matmul(qf, np.transpose(rf))
    
    return cos


def L2(f1, f2, _ord_ = 2):
    
    F1 = fix_dims(f1)
    F2 = fix_dims(f2)  
    
    if not _ord_ is None:
        qf_norms = np.linalg.norm(F1, ord = _ord_, axis=1)
        qf = F1 / qf_norms[:, np.newaxis]
        
        rf_norms = np.linalg.norm(F2, ord = _ord_, axis=1)
        rf = F2 / rf_norms[:, np.newaxis]

    else:
        qf = F1
        rf = F2     
    
    L2 = np.zeros(shape = (np.shape(qf)[0], np.shape(rf)[0]))
    
    for q in range(np.shape(qf)[0]):
        for r in  range(np.shape(rf)[0]):
            diff =  np.linalg.norm((qf[q] - rf[r]), ord = 2)
            L2[q,r] = diff
            
    return L2



def distance(feat1, feat2, norm = True, dist_type="COS"):
    
    if norm:
        ord_ = 2
    else:
        ord_ = None
    
    if dist_type == "L2":
        return L2(feat1, feat2, ord_)
    
    elif dist_type == "COS":
        return COS(feat1, feat2, ord_)
    
    else:
        print("[]: Distance unavailable".format(dist_type))


def get_matches(query_feat, query_labels, ref_feat, ref_labels, top = 1, norm = True, dist_type="COS"):
    
    query_feat_ = np.array(query_feat)
    ref_feat_ = np.array(ref_feat)
    
    simil = distance(query_feat_, ref_feat_, norm, dist_type)
    
    if dist_type == "COS":
        #place sorting here. The sort order might be different for other similarity/distance funtions
        idx = np.argsort(simil, axis = 1)
        #For COS: higher means more similar. As argsort sort in ASCENDING order, we need to flip the indices
        idx = idx[:,::-1]
        
    elif dist_type == "L2":
        #place sorting here. The sort order might be different for other similarity/distance funzions
        idx = np.argsort(simil, axis = 1)
        #For DIST: lower means more similar. As argsort sort in ASCENDING order, thus there is no need to flip the indices
        #idx = idx[:,::-1]
    else:
        print("[]: Distance unavailable".format(dist_type))
        
    
    matches = {}
    coefficients = {}
    
    assert len(query_labels) == np.shape(simil)[0]
    
    for i in range(len(query_labels)):
        label = query_labels[i]
        match_labels = [ref_labels[j] for j in idx[i,:]]
        #top = None => includes all the labels
        matches[label] = match_labels[0:top]
        
        values = simil[i]
        values = values[idx[i,:]]
        coefficients[label] = values[0:top]
    
    return matches, coefficients    


#this works on a very specific file format
def load_gt_csv(filename, how_many = 1):
    
    gt_matches = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if not line.startswith('#'):
                if line.startswith('M'):
                    pass
                if line.startswith('D'):
                    pass
                if line.startswith('N'):
                    tokens = line.split(';')
                    nTM = int(tokens[1]);
                    label = tokens[2]
                    matches = tokens[3:]
                    
                    if nTM >= how_many:
                        gt_matches[label] = matches
                        
    return gt_matches

if __name__ == '__main__':
    pass