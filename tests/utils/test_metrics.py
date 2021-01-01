# metrics has circular dependency on general
# from utils.metrics import fitness
import pytest


class TestMetrics:
    def test_fitness(self):
        pass

    def test_ap_per_class(self):
        pass

    def test_compute_ap(self):
        pass

    def test_plot_pr_curve(self):
        pass


class TestMetricsConfusionMatrix:
    def test_process_batch(self):
        pass

    def test_matrix(self):
        pass

    def test_plot(self):
        pass

    def test_print(self):
        pass
