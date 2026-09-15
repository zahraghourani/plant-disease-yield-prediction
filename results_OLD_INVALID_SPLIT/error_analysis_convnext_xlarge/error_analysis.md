# Error Analysis: ConvNeXtXLarge

This section analyzes failure cases for the best CNN model, ConvNeXtXLarge. The saved confusion matrix is available at:

`results\figures\ConvNeXtXLarge_confusion_matrix.png`

The automated scan evaluated 1200 images and found 96 incorrect predictions, giving an approximate scanned accuracy of 92.00%.

## Most Common Confusion Cases

- `Rice___Healthy` confused as `Rice___Hispa`: 61 examples
- `Rice___Brown_Spot` confused as `Rice___Leaf_Blast`: 10 examples
- `Rice___Leaf_Blast` confused as `Rice___Hispa`: 10 examples
- `Rice___Hispa` confused as `Rice___Healthy`: 4 examples
- `Rice___Healthy` confused as `Rice___Brown_Spot`: 4 examples

## Example Failure Images

- `failure_example_1.png`: true `Rice___Healthy`, predicted `Rice___Hispa` with 100.0% confidence. Brightness=200.2, contrast=53.7.
- `failure_example_2.png`: true `Rice___Brown_Spot`, predicted `Rice___Leaf_Blast` with 97.4% confidence. Brightness=205.7, contrast=46.4.
- `failure_example_3.png`: true `Rice___Leaf_Blast`, predicted `Rice___Hispa` with 100.0% confidence. Brightness=211.0, contrast=49.7.

## Short Discussion

The dominant errors in this scan are within visually similar rice categories, especially healthy rice leaves predicted as `Rice___Hispa`. This suggests that the model sometimes responds to local texture, leaf edges, lighting variation, or small blemishes as if they were disease symptoms. These are plausible failure modes for leaf-disease classification because many classes share similar green leaf backgrounds while the disease cues may be small, sparse, or affected by illumination.

The selected failure examples show cases where the model is confident despite being incorrect. This is important for deployment: high softmax confidence should not be interpreted as a guarantee of correctness. In practice, the interface should show confidence and allow a human user to override the detected crop/disease when the visual evidence is ambiguous. Additional training data with more lighting conditions, healthy leaves, and visually similar rice disease categories would likely reduce these confusions.
