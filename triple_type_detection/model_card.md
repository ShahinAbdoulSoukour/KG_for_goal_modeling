# Fine-tuned RoBERTa Model for Goal Type Classification

## Model Description
This model is a fine-tuned version of RoBERTa-large for classifying verb-context pairs into one of four goal types: **ACHIEVE, MAINTAIN, AVOID, CEASE**.

## Training Data
The model was trained on a dataset containing verbs and their corresponding contexts, labeled according to their goal types.

## Performance Metrics
- **Accuracy**: 0.8333333333333334
- **Precision**: 0.8776223776223776
- **Recall**: 0.8333333333333334
- **F1-score**: 0.8295454545454546

## Test Set Performance
- **Test Accuracy**: 0.9166666666666666
- **Test Precision**: 0.9194444444444445
- **Test Recall**: 0.9166666666666666
- **Test F1-score**: 0.9165806673546611

## Confusion Matrix
See `confusion_matrix.png` for a visual representation of the model's classification performance.

## Loss Plot
See `loss_plot.png` for training and evaluation loss trends over epochs.    