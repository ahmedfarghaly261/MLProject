from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np


def evaluate(model, Y_test, Y_pred, X_test):

    eval_info = { "auc_curve_info": { "Y_test_bin": None, "Y_score": None } }

    eval_info["acc_sc"] = round(accuracy_score(Y_test, Y_pred), 2)
    
    eval_info["cm"] = confusion_matrix(Y_test, Y_pred)

    eval_info["precision"] = round(precision_score(Y_test, Y_pred, average="micro"), 2)
    
    eval_info["recall"] = round(recall_score(Y_test, Y_pred, average="micro"), 2)
    


    eval_info["auc_curve_info"]["Y_score"] = y_score = np.array(model.predict_proba(X_test))

    eval_info["auc_curve_info"]["Y_test_bin"] = y_test_bin = label_binarize(Y_test, classes=np.unique(np.array(Y_test))) 

   
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())

    eval_info["AUC"] = round(roc_auc_score(y_test_bin, y_score, multi_class="ovr"), 2)

    return eval_info


# np.unique(np.array(Y_image_test))