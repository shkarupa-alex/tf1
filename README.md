# tf1

F1-score metrics for classification models in TensorFlow.
There are 3 average modes provided:
- binary
- macro
- micro

## Usage

```python
from tf1 import f1_binary

# use f1_binary as any other metric from tf.metrics.*
```

Note, that due to streaming nature of metric computation process,
"macro" and "micro" average metrics should know total number of
classes. Use them as follows:

```python
from tf1 import f1_macro, f1_micro

def my_task_f1_macro(
    labels, predictions, num_classes, weights=None,
    metrics_collections=None, updates_collections=None,
    name=None):

    return f1_macro(
        labels=labels,
        predictions=predictions,
        num_classes=123,  # Required
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=name
    )

# use my_task_f1_macro as any other metric from tf.metrics.*
```
