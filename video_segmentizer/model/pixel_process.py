from .iid_gaussian import IIDGaussian
import numpy as np

GMM_INITIAL_WEIGHT = 0.2
GMM_INITIAL_VARIANCE = 100


class PixelProcess:

    _n_clusters = None
    _mixture = []

    def __init__(self, n_clusters):
        self._n_clusters = n_clusters

    def _sort_mixture(self):
        self._mixture.sort(key=lambda x: x[0]/np.sqrt(x[1].get_variance()))

    def _get_background_distributions(self, t=0.8):

        sum_weight = 0.0
        background_distributions = []

        for weight, gaussian in self._mixture:
            sum_weight += weight
            background_distributions.append(gaussian)

            if sum_weight > t:
                return background_distributions

        return []

    def _get_first_best_matching_gaussian(self, x):
        return next((i for i, (w, gaus) in enumerate(self._mixture) if gaus.mahalanobis_distance_between(x)), None)

    def _normalize_weights(self):
        total_weight = sum(list(zip(*self._mixture))[0])
        self._mixture = [(w/total_weight, gaus) for w, gaus in self._mixture]

    def fit(self, x, lr=0.75):

        if len(self._mixture) < self._n_clusters:
            self._mixture.append((GMM_INITIAL_WEIGHT, IIDGaussian(x, GMM_INITIAL_VARIANCE, 3)))

        else:

            # Find first matching Gaussian
            best_matching_gaussian_idx = self._get_first_best_matching_gaussian(x)

            if best_matching_gaussian_idx is None:
                self._mixture[self._n_clusters - 1] = (GMM_INITIAL_WEIGHT, IIDGaussian(x, GMM_INITIAL_VARIANCE, 3))

            else:

                best_matching_weight, best_matching_gaussian = self._mixture[best_matching_gaussian_idx]
                best_matching_gaussian.partial_fit(x)

                for i, (w, gaus) in enumerate(self._mixture):

                    if best_matching_gaussian_idx == i:
                        new_weight = (1-lr) * w + lr
                    else:
                        new_weight = (1 - lr) * w

                    self._mixture[i] = (new_weight, gaus)

        # Normalize and ensure correct order
        self._normalize_weights()
        self._sort_mixture()

    def is_background_pixel(self, x):

        background_dist = self._get_background_distributions()

        for gaus in background_dist:
            if gaus.mahalanobis_distance_between(x) < 2.5:
                return True

        return False
