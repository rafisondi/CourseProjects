import warnings
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

domain = np.array([[0, 5]])


""" Solution """
import math
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning

np.random.seed(40)
v_low_limit = 1.2

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        # TODO: enter your code here

        self.domain = np.squeeze(domain)
        f_noise_std = 0.15
        self.kernel_f = 0.5 * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(
            noise_level=f_noise_std**2
        )
        v_noise_std = 0.0001
        # self.kernel_v = math.sqrt(2) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(
        #     noise_level=v_noise_std**2
        # )
        self.kernel_v = ConstantKernel(1.5) * math.sqrt(2) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(
            noise_level=v_noise_std**2
        )         
        self.xs = []
        self.fs = []
        self.vs = []
        self.acq_func = GaussianProcessRegressor(kernel=self.kernel_f, random_state=0)
        self.gpr_for_speed = GaussianProcessRegressor(
            kernel=self.kernel_v, random_state=110
        )
        self.recommendations = []
        self.acq_func_values_at_recommendations = []
        # self.prior_mean_v = 1.5
        self.prior_mean_v = 0
        
    def calc_speed_margin(self, mu_v, sigma_v):
        """Ensures that lower bound on speed from GP is higher than the speed limit"""
        # TODO: some cases in checker_client.py have 20 unsafe evals, while others have zero
        margin = (mu_v + self.prior_mean_v - sigma_v * 3) - v_low_limit
        return margin

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        initial_guess = self.optimize_acquisition_function()
        safe_sample = self.ensure_safe_sample(initial_guess)
        self.acq_func_values_at_recommendations.append(
            self.acquisition_function(safe_sample)
        )
        self.recommendations.append(safe_sample)
        return safe_sample

    def ensure_safe_sample(self, initial_guess):
        # TODO: other ways to shift sample if its speed margin is not big enough?
        safe_sample = np.squeeze(initial_guess)
        patience = 5
        mu_v, sigma_v = self.eval_gpr_for_speed(initial_guess)
        speed_margin = self.calc_speed_margin(mu_v, sigma_v)
        speed_satisfied_cond = speed_margin > 0
        while not speed_satisfied_cond:
            # TODO: how to adapt safe_sample to fall into the safe interval?
            # print(f"Fixing mu_v, sigma_v from: {mu_v, sigma_v}")
            # safe_sample /= 2
            safe_sample_left = max(safe_sample - safe_sample / 4, self.domain[0])
            mu_v_left, sigma_v_left = self.eval_gpr_for_speed(safe_sample_left)
            safe_sample_right = min(safe_sample + safe_sample / 4, self.domain[-1])
            mu_v_right, sigma_v_right = self.eval_gpr_for_speed(safe_sample_right)
            left_speed_margin = self.calc_speed_margin(mu_v_left, sigma_v_left)
            right_speed_margin = self.calc_speed_margin(mu_v_right, sigma_v_right)
            if left_speed_margin > speed_margin:
                mu_v, sigma_v = mu_v_left, sigma_v_left
                safe_sample = safe_sample_left
                # print("fallback to the left")
            elif right_speed_margin > speed_margin:
                mu_v, sigma_v = mu_v_right, sigma_v_right
                safe_sample = safe_sample_right
                # print("fallback to the right")
            else:
                # print("no improvement on both sides. breaking")
                break
            speed_satisfied_cond = self.calc_speed_margin(mu_v, sigma_v) > 0
            # mu_v, sigma_v = self.eval_gpr_for_speed(safe_sample)
            # speed_satisfied_cond = self.calc_speed_margin(mu_v, sigma_v) > 0
            # print(f"to: {mu_v, sigma_v}")
            patience -= 1
            if patience == 0:
                print("Patience is up. Falling back to init guess")
                break
        # safe_sample = initial_guess if patience == 0 else safe_sample

        return np.atleast_2d(safe_sample)

    def eval_gpr_for_speed(self, x):
        """
        Evaluates GP for speed given a sample
        """
        mu_v, sigma_v = self.gpr_for_speed.predict(
            np.atleast_2d(x).reshape(-1, 1), return_std=True
        )
        return mu_v, sigma_v

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            mu_v, sigma_v = self.eval_gpr_for_speed(x)
            acq_func_val = self.acquisition_function(x)
            # TODO: how to derive penalty?
            penalty = 0 if self.calc_speed_margin(mu_v, sigma_v) > 0 else 1
            return -(acq_func_val - penalty)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
                domain.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x, return_std=False):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # UCB acq
        # TODO: fine-tune "eta"
        eta = 1
        mu_f, sigma_f = self.acq_func.predict(
            np.atleast_2d(x).reshape(-1, 1), return_std=True
        )

        func_val = mu_f.flatten() + eta * sigma_f
        if return_std:
            return func_val, sigma_f
        return func_val

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.xs.append(np.squeeze(x))
        self.fs.append(np.squeeze(f))
        self.vs.append(
            np.squeeze(v - self.prior_mean_v)
        )  # since the speed GP works for mean=zero
        self.acq_func.fit(
            np.atleast_2d(self.xs).reshape(-1, 1), np.atleast_2d(self.fs).reshape(-1, 1)
        )
        self.gpr_for_speed.fit(
            np.atleast_2d(self.xs).reshape(-1, 1), np.atleast_2d(self.vs).reshape(-1, 1)
        )

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # return self.recommendations[
        #     np.argmax(self.acq_func_values_at_recommendations)
        # ].reshape(1, domain.shape[0])
    
        fs_ind_descending = np.argsort(self.fs)[::-1]
        x_opt_ind = 0
        for ind in fs_ind_descending:
            if self.vs[ind] >= v_low_limit:
                x_opt_ind = ind
                break
            else:
                x_opt_ind = fs_ind_descending[0] # return xs with max fs if all violates v_low_limit
                
        return self.xs[x_opt_ind]    

    def plot_results(self):
        import matplotlib.pyplot as plt

        plt.scatter(self.xs, self.fs)
        Xsamples = np.asarray(np.arange(0, math.pi, 0.01))
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        mu, std = self.acquisition_function(Xsamples, return_std=True)
        colors = plt.cm.rainbow(np.linspace(0, 1, 10))
        plt.plot(Xsamples, mu, label="Mean prediction", color=colors[0])
        plt.fill_between(
            Xsamples.ravel(),
            mu - 1.96 * std,
            mu + 1.96 * std,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("Gaussian process regression on noise-free dataset")
        plt.show()


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    return np.sin(x)
    # mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    # return -np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    n_dim = 1
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    # num_iters = 20
    num_iters = 25
    for j in range(num_iters):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {domain.shape[0]})"
        )

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), (
        f"The function get solution must return a numpy array of shape ("
        f"1, {domain.shape[0]})"
    )
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the "
        f"domain, {solution} returned instead"
    )

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = 0 - f(solution)

    print(
        f"\nOptimal value: pi/2\nf(optimal)={f(math.pi/2)}\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret {regret}"
    )

    agent.plot_results()


if __name__ == "__main__":
    main()
