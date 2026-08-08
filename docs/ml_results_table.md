# Classical ML Baseline Results

All results: 5-fold nested cross-validation, mean across outer folds. "Combat" = ComBat site harmonization (fit per outer-fold train split, applied to test split, leak-free). Sorted by test accuracy, descending.


| Algorithm | Feature Set | ComBat | N Features | Accuracy | AUC | F1 | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|---|---|---|
| elasticnet | gpcc_alff | Yes | 4275 | 0.682 ± 0.029 | 0.747 | 0.654 | 0.633 | 0.726 | 0.677 |
| gradient_boosting | gpcc_alff | Yes | 4275 | 0.676 ± 0.018 | 0.733 | 0.657 | 0.653 | 0.697 | 0.663 |
| gradient_boosting | global_pcc | Yes |  | 0.670 ± 0.032 | 0.733 | 0.646 | 0.633 | 0.704 | 0.661 |
| random_forest | gpcc_lpcc | Yes | 12015 | 0.669 ± 0.033 | 0.714 | 0.615 | 0.558 | 0.770 | 0.687 |
| rbf_svm | global_pcc | Yes |  | 0.668 ± 0.016 | 0.732 | 0.639 | 0.618 | 0.715 | 0.662 |
| random_forest | gpcc_alff | Yes | 4275 | 0.668 ± 0.021 | 0.718 | 0.625 | 0.582 | 0.747 | 0.677 |
| mlp | global_pcc | Yes |  | 0.666 ± 0.026 | 0.717 | 0.650 | 0.651 | 0.681 | 0.650 |
| elasticnet | global_pcc | Yes |  | 0.666 ± 0.025 | 0.737 | 0.637 | 0.615 | 0.713 | 0.660 |
| random_forest | global_pcc | Yes |  | 0.666 ± 0.034 | 0.731 | 0.619 | 0.569 | 0.754 | 0.679 |
| elasticnet | global_pcc | No |  | 0.665 ± 0.019 | 0.711 | 0.637 | 0.618 | 0.709 | 0.658 |
| rbf_svm | all3 | Yes | 12285 | 0.665 ± 0.028 | 0.727 | 0.633 | 0.607 | 0.719 | 0.662 |
| mlp | gpcc_alff | Yes | 4275 | 0.665 ± 0.032 | 0.724 | 0.640 | 0.629 | 0.699 | 0.654 |
| rbf_svm | lpcc_alff | Yes | 8280 | 0.664 ± 0.029 | 0.712 | 0.628 | 0.598 | 0.725 | 0.663 |
| rbf_svm | gpcc_lpcc | Yes | 12015 | 0.662 ± 0.025 | 0.725 | 0.633 | 0.613 | 0.707 | 0.655 |
| rbf_svm | gpcc_alff | Yes | 4275 | 0.661 ± 0.012 | 0.730 | 0.634 | 0.618 | 0.701 | 0.652 |
| mlp | all3 | Yes | 12285 | 0.658 ± 0.022 | 0.712 | 0.647 | 0.662 | 0.655 | 0.634 |
| gradient_boosting | global_pcc | No |  | 0.655 ± 0.038 | 0.706 | 0.616 | 0.585 | 0.718 | 0.652 |
| random_forest | all3 | Yes | 12285 | 0.655 ± 0.034 | 0.717 | 0.600 | 0.545 | 0.754 | 0.669 |
| mlp | gpcc_lpcc | Yes | 12015 | 0.654 ± 0.006 | 0.718 | 0.627 | 0.611 | 0.693 | 0.644 |
| rbf_svm | global_pcc | No |  | 0.651 ± 0.004 | 0.725 | 0.623 | 0.607 | 0.691 | 0.641 |
| elasticnet | local_pcc | Yes |  | 0.651 ± 0.028 | 0.700 | 0.628 | 0.620 | 0.679 | 0.640 |
| elasticnet | gpcc_lpcc | Yes | 12015 | 0.649 ± 0.039 | 0.711 | 0.620 | 0.607 | 0.687 | 0.638 |
| mlp | global_pcc | No |  | 0.649 ± 0.021 | 0.707 | 0.625 | 0.618 | 0.677 | 0.634 |
| rbf_svm | local_pcc | No |  | 0.647 ± 0.022 | 0.689 | 0.612 | 0.587 | 0.703 | 0.642 |
| linear_svm | gpcc_alff | Yes | 4275 | 0.647 ± 0.023 | 0.689 | 0.630 | 0.631 | 0.663 | 0.629 |
| rbf_svm | local_pcc | Yes |  | 0.646 ± 0.041 | 0.703 | 0.611 | 0.582 | 0.705 | 0.644 |
| mlp | lpcc_alff | Yes | 8280 | 0.645 ± 0.026 | 0.703 | 0.621 | 0.613 | 0.675 | 0.632 |
| elasticnet | all3 | Yes | 12285 | 0.645 ± 0.038 | 0.714 | 0.621 | 0.613 | 0.675 | 0.631 |
| linear_svm | gpcc_lpcc | Yes | 12015 | 0.644 ± 0.022 | 0.705 | 0.632 | 0.642 | 0.647 | 0.624 |
| gradient_boosting | gpcc_lpcc | Yes | 12015 | 0.644 ± 0.034 | 0.698 | 0.609 | 0.582 | 0.700 | 0.639 |
| linear_svm | global_pcc | No |  | 0.642 ± 0.036 | 0.689 | 0.628 | 0.635 | 0.649 | 0.622 |
| random_forest | global_pcc | No |  | 0.642 ± 0.029 | 0.694 | 0.591 | 0.545 | 0.730 | 0.646 |
| mlp | local_pcc | Yes |  | 0.640 ± 0.035 | 0.702 | 0.617 | 0.609 | 0.668 | 0.629 |
| linear_svm | global_pcc | Yes |  | 0.639 ± 0.026 | 0.691 | 0.624 | 0.631 | 0.647 | 0.618 |
| random_forest | local_pcc | Yes |  | 0.639 ± 0.040 | 0.683 | 0.553 | 0.475 | 0.789 | 0.668 |
| elasticnet | lpcc_alff | Yes | 8280 | 0.638 ± 0.040 | 0.702 | 0.609 | 0.593 | 0.679 | 0.629 |
| random_forest | lpcc_alff | Yes | 8280 | 0.638 ± 0.037 | 0.694 | 0.570 | 0.505 | 0.758 | 0.655 |
| elasticnet | local_pcc | No |  | 0.637 ± 0.031 | 0.682 | 0.609 | 0.598 | 0.673 | 0.625 |
| linear_svm | local_pcc | Yes |  | 0.635 ± 0.017 | 0.684 | 0.619 | 0.624 | 0.645 | 0.617 |
| linear_svm | all3 | Yes | 12285 | 0.634 ± 0.016 | 0.701 | 0.618 | 0.624 | 0.643 | 0.614 |
| gradient_boosting | all3 | Yes | 12285 | 0.632 ± 0.030 | 0.690 | 0.596 | 0.571 | 0.686 | 0.624 |
| gradient_boosting | lpcc_alff | Yes | 8280 | 0.631 ± 0.023 | 0.678 | 0.588 | 0.554 | 0.700 | 0.627 |
| linear_svm | local_pcc | No |  | 0.630 ± 0.021 | 0.679 | 0.616 | 0.624 | 0.635 | 0.611 |
| linear_svm | lpcc_alff | Yes | 8280 | 0.624 ± 0.023 | 0.679 | 0.612 | 0.624 | 0.625 | 0.603 |
| mlp | local_pcc | No |  | 0.623 ± 0.025 | 0.676 | 0.591 | 0.574 | 0.669 | 0.612 |
| random_forest | local_pcc | No |  | 0.612 ± 0.016 | 0.666 | 0.519 | 0.442 | 0.766 | 0.633 |
| gradient_boosting | local_pcc | Yes |  | 0.611 ± 0.042 | 0.668 | 0.576 | 0.556 | 0.661 | 0.598 |
| gradient_boosting | local_pcc | No |  | 0.606 ± 0.019 | 0.644 | 0.562 | 0.532 | 0.673 | 0.597 |
| mlp | alff | No |  | 0.590 ± 0.025 | 0.609 | 0.553 | 0.534 | 0.641 | 0.575 |
| linear_svm | alff | No |  | 0.586 ± 0.022 | 0.598 | 0.548 | 0.530 | 0.637 | 0.570 |
| elasticnet | alff | No |  | 0.583 ± 0.022 | 0.616 | 0.545 | 0.525 | 0.635 | 0.568 |
| knn | local_pcc | Yes |  | 0.574 ± 0.036 | 0.590 | 0.497 | 0.444 | 0.692 | 0.568 |
| random_forest | alff | No |  | 0.573 ± 0.038 | 0.593 | 0.499 | 0.446 | 0.689 | 0.571 |
| rbf_svm | alff | No |  | 0.572 ± 0.029 | 0.605 | 0.514 | 0.477 | 0.659 | 0.559 |
| knn | gpcc_lpcc | Yes | 12015 | 0.572 ± 0.019 | 0.576 | 0.482 | 0.420 | 0.711 | 0.568 |
| knn | local_pcc | No |  | 0.562 ± 0.026 | 0.571 | 0.450 | 0.378 | 0.728 | 0.559 |
| gradient_boosting | alff | No |  | 0.559 ± 0.030 | 0.575 | 0.477 | 0.424 | 0.681 | 0.551 |
| knn | lpcc_alff | Yes | 8280 | 0.554 ± 0.028 | 0.571 | 0.452 | 0.387 | 0.706 | 0.548 |
| knn | global_pcc | No |  | 0.552 ± 0.028 | 0.573 | 0.440 | 0.371 | 0.717 | 0.546 |
| knn | all3 | Yes | 12285 | 0.549 ± 0.007 | 0.564 | 0.443 | 0.380 | 0.703 | 0.538 |
| knn | alff | No |  | 0.547 ± 0.022 | 0.563 | 0.440 | 0.376 | 0.703 | 0.534 |
| knn | gpcc_alff | Yes | 4275 | 0.544 ± 0.024 | 0.569 | 0.478 | 0.442 | 0.637 | 0.525 |
| knn | global_pcc | Yes |  | 0.535 ± 0.043 | 0.556 | 0.489 | 0.468 | 0.595 | 0.513 |
