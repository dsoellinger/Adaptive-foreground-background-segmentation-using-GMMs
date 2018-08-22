from .iid_gaussian import IIDGaussian


class GMM:

    _n_clusters = None
    _mixture = []

    def __init__(self, n_clusters):
        self._n_clusters = n_clusters

    def _sort_gaussians(self):
        pass

    def _get_background_distributions(self, t):

        sum_weight = 0.0
        background_distributions = []

        for weight, gaussian in self._mixture:
            sum_weight += weight
            background_distributions.append(gaussian)

            if sum_weight > t:
                return background_distributions

    def _normalize_weights(self):
        total_weights = sum(list(map(lambda x: x[0], self._mixture)))

        for i in range(len(self._mixture)):
            self._mixture[i] = (self._mixture[i][0]/total_weights, self._mixture[i][1])

    def partial_fit(self, x, lr=0.75):

        best_matching_gaussian_idx = None

        for i, tmp in enumerate(self._mixture):
            weight, gaussian = tmp
            if gaussian.mahalanobis_distance_between(x) < 2.5:
                best_matching_gaussian_idx = i
                break

        if best_matching_gaussian_idx is None:

            if len(self._mixture) < self._n_clusters:
                self._mixture.append((0.2, IIDGaussian(x, 100, 3)))
            else:
                self._mixture[self._n_clusters - 1] = (0.2, IIDGaussian(x, 100, 3) )

        else:

            best_matching_weight, best_matching_gaussian = self._mixture[best_matching_gaussian_idx]
            best_matching_gaussian.partial_fit(x)

            for i in range(len(self._mixture)):

                weight, gaussian = self._mixture[i]

                if best_matching_gaussian_idx == i:
                    new_weight = (1-lr) * weight + lr
                else:
                    new_weight = (1 - lr) * weight

                self._mixture[i] = (new_weight, gaussian)

        self._normalize_weights()

        # Ensure that the gaussians in descending order (w/sigma)
        self._sort_gaussians()



