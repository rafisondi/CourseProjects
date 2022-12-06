import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

# Build regions for local GPs based on rough clusters found in the 3D (lat, lon, pm25) plot
(lat_low_1, lat_high_1, lon_low_1, lon_high_1) = (0, 1.0, 0.0, 0.2)
(lat_low_2, lat_high_2, lon_low_2, lon_high_2) = (0, 1.0, 0.2, 0.6)
(lat_low_3, lat_high_3, lon_low_3, lon_high_3) = (0, 1.0, 0.6, 1.0)
region_1 = (lat_low_1, lat_high_1, lon_low_1, lon_high_1)
region_2 = (lat_low_2, lat_high_2, lon_low_2, lon_high_2)
region_3 = (lat_low_3, lat_high_3, lon_low_3, lon_high_3)


def shift_preds(preds, val_std):
    """
    Modifies raw predictions to account for the case of predicted < true PM.
    Coefficients were obtained using Bayesian optimization.
    More details could be found in the fit_tuning_errors.ipynb notebook

    Args:
        preds: raw predictions
        val_std: std from GP

    Returns:
        shifted predictions
    """
    val_pred = preds.copy()
    coef_avg_std = 0.5185760986204269
    coef_big_std = 0.2443434966375043
    coef_small_std = 1.0
    q_big_unc_mask = 0.8
    q_avg_unc_mask_low = 0.2
    q_avg_unc_mask_high = 0.8
    q_small_unc_mask = 0.2
    big_unc_mask = val_std > np.quantile(val_std, q_big_unc_mask)
    avg_unc_mask = (val_std > np.quantile(val_std, q_avg_unc_mask_low)) & (
        val_std < np.quantile(val_std, q_avg_unc_mask_high)
    )
    mild_unc_mask = val_std < np.quantile(val_std, q_small_unc_mask)
    val_pred[big_unc_mask] += coef_big_std * val_std[big_unc_mask]
    val_pred[avg_unc_mask] += coef_avg_std * val_std[avg_unc_mask]
    val_pred[mild_unc_mask] += coef_small_std * val_std[mild_unc_mask]
    return val_pred


def get_region_mask(data, region_bounds):
    """
    Subsample cluster from data based on lat and lon

    Args:
        data: input features
        region_bounds: tuple with (lat_low, lat_high, lon_low, lon_high) values that define a region

    Returns:
        subsample of data with eligible lat and lon values
    """
    return (
        (region_bounds[0] <= data[:, 0])
        & (data[:, 0] < region_bounds[1])
        & (region_bounds[2] <= data[:, 1])
        & (data[:, 1] < region_bounds[3])
    )


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        np.random.seed(100)
        # TODO: Add custom initialization for your model here if necessary
        self.kernel1 = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.kernel2 = RBF(
            length_scale=1.0, length_scale_bounds=(1e-3, 1e3)
        ) + WhiteKernel(noise_level=5e-2)
        self.region_to_gp = {
            # (lat_low, lat_high, lon_low, lon_high): gpr_for_this_region
        }

    def make_predictions(
        self, test_features: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        region_masks_val = []
        gp_mean = []
        gp_std = []
        preds_val = []
        preds_val_stats = []
        for region in [
            region_1,
            region_2,
            region_3,
        ]:
            region_mask_val = get_region_mask(test_features, region)
            features_val = test_features[region_mask_val]
            gpr = self.region_to_gp[region]
            val_pred_mean, val_pred_std = gpr.predict(features_val, return_std=True)
            val_pred = val_pred_mean
            region_masks_val.append(region_mask_val)
            preds_val.append(val_pred)
            preds_val_stats.append([val_pred_mean, val_pred_std])

        # TODO: Use the GP posterior to form your predictions here
        predictions = np.zeros(shape=(len(test_features), 1))
        preds_val_stats_final = np.zeros(shape=(len(predictions), 2))
        for region_mask, pred_val, pred_val_stats in zip(
            region_masks_val, preds_val, preds_val_stats
        ):
            predictions[region_mask] = pred_val[:, np.newaxis]
            preds_val_stats_final[region_mask] = np.array(pred_val_stats).T

        gp_mean = np.array(gp_mean)
        gp_std = np.array(gp_std)
        predictions = np.squeeze(predictions)
        val_mean, val_std = (
            preds_val_stats_final[:, 0].copy(),
            preds_val_stats_final[:, 1].copy(),
        )
        predictions = shift_preds(predictions, val_std)
        return predictions, val_mean, val_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        from sklearn import model_selection

        take_n_samples = 5500
        train_x = train_features
        train_y = train_GT
        train_x, val_x, train_y, val_y = model_selection.train_test_split(
            train_features, train_GT, test_size=0.2
        )

        gprs = []
        region_masks_val = []
        preds_val = []
        for region in [
            region_1,
            region_2,
            region_3,
        ]:
            region_mask_train = get_region_mask(train_x, region)
            features_train = train_x[region_mask_train][:take_n_samples]
            labels_train = train_y[region_mask_train][:take_n_samples]
            gpr = GaussianProcessRegressor(
                kernel=Sum(self.kernel1, self.kernel2), alpha=1e-3, normalize_y=True
            )
            gpr.fit(features_train, labels_train)
            gprs.append(gpr)

            region_mask_val = get_region_mask(val_x, region)
            features_val = val_x[region_mask_val]
            labels_val = val_y[region_mask_val]
            val_pred_mean, val_pred_std = gpr.predict(features_val, return_std=True)
            val_pred = val_pred_mean
            val_pred = shift_preds(val_pred, val_pred_std)
            print(cost_function(labels_val.squeeze(), val_pred.squeeze()))
            region_masks_val.append(region_mask_val)
            preds_val.append(val_pred)
            self.region_to_gp[region] = gpr


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert (
        ground_truth.ndim == 1
        and predictions.ndim == 1
        and ground_truth.shape == predictions.shape
    )

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = predictions >= 1.2 * ground_truth
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = "/results"):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print("Performing extended evaluation")
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle("Extended visualization of task 1")

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS)
        / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    )
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title("Predictions")
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection="3d")
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False,
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title("GP means, colors are GP stddev")

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title("GP estimated stddev")
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, "extended_evaluation.pdf")
    fig.savefig(figure_path)
    print(f"Saved extended evaluation to {figure_path}")

    plt.show()


def main():
    base_dir = "/media/master/MyPassport/master_studies/first_semester/probabalistic_ai/assignments/assignment1"
    # Load the training dateset and test features
    train_features = np.loadtxt(f"{base_dir}/train_x.csv", delimiter=",", skiprows=1)
    train_GT = np.loadtxt(f"{base_dir}/train_y.csv", delimiter=",", skiprows=1)
    test_features = np.loadtxt(f"{base_dir}/test_x.csv", delimiter=",", skiprows=1)

    # Fit the model
    print("Fitting model")
    model = Model()
    model.fitting_model(train_GT, train_features)

    # Predict on the test features
    print("Predicting on test features")
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir=".")


if __name__ == "__main__":
    main()
