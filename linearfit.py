import argparse
from scipy.optimize import minimize
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)


# d/dw_i |sum(x_j w_j) - y| = x_i sgn(sum(x_j w_j) - y)

def l1(params, data, weights, labels, jac=True):
    prediction = np.dot(data, params.reshape(-1, 1))
    dist = prediction - labels
    assert dist.shape == (data.shape[0], 1)
    abs_dist = np.abs(dist)
    weighted_sum = (weights * abs_dist).sum()
    if not jac:
        return weighted_sum
    sign_dist = np.sign(dist)
    return weighted_sum, (weights.reshape(-1, 1) * (sign_dist * data)).sum(axis=0)


def l2(params, data, weights, labels):
    prediction = np.dot(data, params.reshape(-1, 1))
    dist = prediction - labels
    return (weights * dist**2).sum()


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    n = 100
    d = 5
    data = np.random.random((n, d))
    weights = np.random.random(n) + 1
    weights /= weights.sum()
    real_params = np.random.random((d, 1))
    real_labels = np.dot(data, real_params)
    assert real_labels.shape == (n, 1)
    noise = np.random.random((n, 1))
    noise_labels = real_labels + noise
    assert noise_labels.shape == (n, 1)

    initial_params = np.zeros(d)
    res = minimize(l1, initial_params, args=(data, weights, noise_labels, False), jac=False, method='SLSQP')
    params_l2, _residuals, _rank, _s = np.linalg.lstsq(
        data * weights.reshape(-1, 1), noise_labels * weights.reshape(-1, 1),
        rcond=None)
    print(res)
    params_l1 = res.x
    l1_l1 = l1(params_l1, data, weights, noise_labels, jac=False)
    l2_l1 = l1(params_l2, data, weights, noise_labels, jac=False)
    l1_l2 = l2(params_l1, data, weights, noise_labels)
    l2_l2 = l2(params_l2, data, weights, noise_labels)
    print('')
    print('L1-param:', params_l1)
    print('L2-param:', params_l2.reshape(-1))
    print('Real param:', real_params.reshape(-1))
    print('')
    print('L1-param L1-score:', l1_l1)
    print('L2-param L1-score:', l2_l1)
    print('Real L1-score:', l1(real_params, data, weights, noise_labels, jac=False))
    print('')
    print('L1-param L2-score:', l1_l2)
    print('L2-param L2-score:', l2_l2)
    print('Real L2-score:', l2(real_params, data, weights, noise_labels))
    assert l1_l1 < l2_l1
    assert l2_l2 < l1_l2


if __name__ == '__main__':
    main()
