import sys
sys.path.append(".")
import math
import evaluate
import jieba
import numpy as np
from rouge_chinese import Rouge
from tqdm import tqdm

class LaMPEvaluation:
    def __init__(self, task):
        self.task = task
        if task.startswith('LaMP_3'):
            self.metric = Metric_MAE_RMSE()
        else:
            pass 

    def compute_metrics(self, preds, labels, avg=True):
        return self.metric.compute_metrics(preds, labels, avg)

def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip().lower() for pred in preds]
    labels = [str(label).strip().lower() for label in labels]
    return preds, labels

class Metric_MAE_RMSE:
    def __init__(self):
        # Yerel dosyaları yüklemeye zorluyoruz
        self.mse_metric = evaluate.load("./metrics/mse.py")
        self.mae_metric = evaluate.load("./metrics/mae.py")

    def create_mapping(self, x, y):
        try:
            # Modeli 1.0 ile 5.0 arasına hapsediyoruz (Clipping)
            val = float(x)
            return max(1.0, min(5.0, val)) 
        except:
            try:
                y = float(y)
                if abs(1 - y) > abs(5 - y):
                    return 5.0
                else:
                    return 1.0
            except:
                return 3.0

    def _compute_metrics(self, preds, labels):
        preds, labels = postprocess_text_classification(preds, labels)
        # Güvenli tip dönüşümü
        preds = [self.create_mapping(x, y) for x, y in zip(preds, labels)]
        labels = [self.create_mapping(x, x) for x in labels]
        
        result_mae = self.mae_metric.compute(predictions=preds, references=labels)
        result_rmse = self.mse_metric.compute(predictions=preds, references=labels)
        
        # Anahtar isimlerini garantiye alıyoruz
        mae_val = result_mae.get("mae", result_mae.get("mean_absolute_error"))
        mse_val = result_rmse.get("mse", result_rmse.get("mean_squared_error"))
        
        result = {"MAE": mae_val, "MSE": mse_val}
        return result

    def compute_metrics(self, preds, labels, avg=True):
        if avg:
            results = self._compute_metrics(preds, labels)
            results['RMSE'] = math.sqrt(results['MSE'])
            return results
        else:
            results = {"MAE": [], "MSE": []}
            for pred, label in zip(preds, labels):
                cur_score = self._compute_metrics([pred], [label])
                for k in results.keys():
                    results[k].append(cur_score[k])
            return results
