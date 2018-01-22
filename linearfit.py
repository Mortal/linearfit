import argparse
from scipy.optimize import minimize
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)


def l1(params, data, noise_labels):
    prediction = np.dot(data, params.reshape(-1, 1))
    dist = prediction - noise_labels
    return np.abs(dist).mean()


def l2(params, data, noise_labels):
    prediction = np.dot(data, params.reshape(-1, 1))
    dist = prediction - noise_labels
    return (dist**2).mean()


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    n = 100
    d = 5
    data = np.random.random((n, d))
    real_params = np.random.random((d, 1))
    real_labels = np.dot(data, real_params)
    assert real_labels.shape == (n, 1)
    noise = np.random.random((n, 1))
    noise_labels = real_labels + noise
    assert noise_labels.shape == (n, 1)

    initial_params = np.zeros(d)
    res = minimize(l1, initial_params, args=(data, noise_labels))
    params_l2, _residuals, _rank, _s = np.linalg.lstsq(data, noise_labels, rcond=None)
    print(res)
    params_l1 = res.x
    print('')
    print('L1-param:', params_l1)
    print('L2-param:', params_l2.reshape(-1))
    print('Real param:', real_params.reshape(-1))
    print('')
    print('L1-param L1-score:', l1(params_l1, data, noise_labels))
    print('L2-param L1-score:', l1(params_l2, data, noise_labels))
    print('Real L1-score:', l1(real_params, data, noise_labels))
    print('')
    print('L1-param L2-score:', l2(params_l1, data, noise_labels))
    print('L2-param L2-score:', l2(params_l2, data, noise_labels))
    print('Real L2-score:', l2(real_params, data, noise_labels))


if __name__ == '__main__':
    main()
