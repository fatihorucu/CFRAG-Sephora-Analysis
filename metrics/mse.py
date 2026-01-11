import datasets
import evaluate
from sklearn.metrics import mean_squared_error

_DESCRIPTION = "Mean Squared Error."
_KWARGS_DESCRIPTION = "Args: predictions, references"
_CITATION = ""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Mse(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("float"),
                "references": datasets.Value("float"),
            }),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"],
        )

    def _compute(self, predictions, references, sample_weight=None, multioutput="uniform_average", squared=True):
        # 'squared' parametresi yeni sklearn sürümlerinde sorun çıkardığı için kullanmıyoruz
        return {"mse": mean_squared_error(references, predictions, sample_weight=sample_weight, multioutput=multioutput)}
