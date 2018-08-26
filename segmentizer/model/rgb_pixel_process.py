from .iid_gaussian import IIDGaussian

GMM_INITIAL_WEIGHT = 1
GMM_INITIAL_VARIANCE = 6.0


class RGBPixelProcess:

    def __init__(self, n_clusters):
        self._n_clusters = n_clusters
        self._mixture = []

    def _sort_mixture(self):
        self._mixture.sort(key=lambda x: x[0]/x[1].get_sigma(), reverse=True)

    def _get_background_distributions(self, t=0.9):

        sum_weight = 0.0
        background_distributions = []

        for weight, gaussian in self._mixture:
            sum_weight += weight
            background_distributions.append(gaussian)
            if sum_weight > t:
                return background_distributions

        return []

    def _get_first_best_matching_gaussian(self, x):

        for i, (w, gaus) in enumerate(self._mixture):
            if gaus.mahalanobis_distance_between(x) < 2.5:
                return i

    def _normalize_weights(self, total_weight):

        self._mixture = [(w/total_weight, gaus) for w, gaus in self._mixture]

    def fit(self, x, lr=0.01):

        if len(self._mixture) < self._n_clusters:
            self._mixture.append((GMM_INITIAL_WEIGHT, IIDGaussian(x, GMM_INITIAL_VARIANCE)))
            self._normalize_weights(total_weight=len(self._mixture))

        else:

            # Find first matching Gaussian
            best_matching_gaussian_idx = self._get_first_best_matching_gaussian(x)

            if best_matching_gaussian_idx is None:
                self._mixture[self._n_clusters - 1] = (GMM_INITIAL_WEIGHT, IIDGaussian(x, GMM_INITIAL_VARIANCE))

            else:

                best_matching_weight, best_matching_gaussian = self._mixture[best_matching_gaussian_idx]

                best_matching_gaussian.partial_fit(x, 0.75)

                total_weight = 0.0

                for i, (w, gaus) in enumerate(self._mixture):

                    if best_matching_gaussian_idx == i:
                        new_weight = (1-lr) * w + lr
                    else:
                        new_weight = (1-lr) * w

                    total_weight += new_weight

                    self._mixture[i] = (new_weight, gaus)

                self._normalize_weights(total_weight)

        self._sort_mixture()

        return self

    def is_background_pixel(self, x):

        background_dist = self._get_background_distributions()
        for gaus in background_dist:
            if gaus.mahalanobis_distance_between(x) < 2.5:
                return True

    def __str__(self):
        s = 'RGBPixelProcess[' + ','.join(map(lambda x: 'w:' + str(x[0]) + ', gaus:' + str(x[1]),self._mixture)) + ']'
        return s
