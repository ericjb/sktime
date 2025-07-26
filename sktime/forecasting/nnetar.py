# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Shallow Neural Net for AutoRegressive AR(p,P) models.

   (following NNETAR in the R package fable by Hyndman et al)

- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, cutoff, _fh,
    _cutoff, _converter_store_y, forecasters_, _tags, _tags_dynamic, _is_vectorized
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None)

Optional implements:
    updating                    - _update(self, y, X=None, update_params=True):
    predicting quantiles        - _predict_quantiles(self, fh, X=None, alpha=None)
    OR predicting intervals     - _predict_interval(self, fh, X=None, coverage=None)
    predicting variance         - _predict_var(self, fh, X=None, cov=False)
    distribution forecast       - _predict_proba(self, fh, X=None)
    fitted parameter inspection - _get_fitted_params()

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""

__author__ = ["ericjb"]

import math

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose._reduce import RecursiveReductionForecaster
from sktime.transformations.series.detrend import Deseasonalizer

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class NNetAR(BaseForecaster):
    """Shallow Neural Net for AutoRegressive AR(p,P) models.

    Motivated by the nnetar() and NNETAR() functions in the
    R packages forecast and fable, respectively.

    The following description is taken from
    https://cran.r-project.org/web/packages/fable/fable.pdf

    "A feed-forward neural network is fitted with lagged values of the
    response as inputs and a single hidden layer with size nodes. The
    inputs are for lags 1 to p, and lags m to mP where m is the seasonal
    period specified.

    If exogenous regressors are provided, its columns are also used as
    inputs. Missing values are currently not supported by this
    model. A total of repeats networks are fitted, each with random
    starting weights. These are then averaged when computing
    forecasts. The network is trained for one-step
    forecasting. Multi-step forecasts are computed recursively.

    For non-seasonal data, the fitted model is denoted as an NNAR(p,k)
    model, where k is the number of hidden nodes. This is analogous to
    an AR(p) model but with non-linear functions. For seasonal data,
    the fitted model is called an NNAR(p,P,k)[m] model, which is
    analogous to an ARIMA(p,0,0)(P,0,0)[m] model but with non-linear
    functions."

    Parameters
    ----------
    ml_reg : class, (default=None)
        An sklearn regressor.
        If None, MLPRegressor from sklearn.neural_network is used.
        In the first implementation, must be set to None.
    p : int, optional (ignored if auto is True)
        The order of the non-seasonal auto-regressive (AR) terms.
        If p is None, an optimal number of lags will be selected for.
    P : int, (default=1, ignored if `period' is None or auto is True)
        The order of the seasonal auto-regressive (SAR) terms.
    period : int, optional (default=None)
        The number of observations in each seasonal period.
        e.g. 12 for monthly time series with seasonality.
    auto : bool, optional (default=False)
        If True, `p' and `P' are ignored, and optimal values for `p' and `P'
        are determined via AIC. For seasonal time series, this will be computed
        on the seasonally adjusted data (via STL decomposition)
    max_p : int, optional (default=2, ignored if auto is False)
        Maximum number of non-seasonal lags.
    max_P : int, optional (default=1, ignored if auto is False)
        Maximum number of seasonal lags.
    X : pd.DataFrame, optional (default=None)
        Exogenous variables
    n_nodes : int, optional
        Number of nodes in the hidden layer; Default is half of the number
        of input nodes (incl X, if given) plus 1
    n_networks : int, (default=20)
        Number of networks to fit with different random starting weights.
        These are then averaged when producing forecasts.
    scale_inputs : bool, (default=True)
        If True, inputs are scaled by subtracting the column means and dividing
        by their respective standard deviations.

    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["ericjb"],
        "maintainers": ["ericjb"],
        "python_version": None,
        "python_dependencies": None,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        ml_reg=None,
        p=None,
        P=1,
        period=None,
        auto=False,
        max_p=2,
        max_P=1,
        X=None,
        n_nodes=None,
        n_networks=20,
        scale_inputs=True,
    ):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.ml_reg = ml_reg
        self.p = p
        self.P = P
        self.period = period
        self.auto = auto
        self.max_p = max_p
        self.max_P = max_P
        self.X = X
        self.n_nodes = n_nodes
        self.n_networks = n_networks
        self.scale_inputs = scale_inputs
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama
        # for estimators, initialize a clone, e.g., self.est_ = est.clone()

        # leave this as is
        super().__init__()

        # parameter checking (including default values and checking for features
        # not yet supported)
        if ml_reg is not None:
            raise ValueError("**NNetAR: not supported**: ml_reg must be None")

        if period is not None:
            if not (isinstance(period, int) and period > 1):
                raise ValueError(
                    "**NNetAR: error**: period must be None or a positive integer > 1"
                )

        if not auto:
            if p is None:
                raise ValueError("**NNetAR: error**: p must be a positive integer")
            self._p = self.p
            self._P = P

        if X is not None:
            raise ValueError("**NNetAR: not supported**: X must be None")

        if n_nodes is None:
            self._n_nodes = math.ceil((self._p + self._P) / 2 + 1)
        else:
            if not (isinstance(n_nodes, int) and n_nodes > 0):
                raise ValueError(
                    "**NNetAR: error**: n_nodes must be a positive integer or None"
                )
            else:
                self._n_nodes = self.n_nodes

        if n_networks is None:
            self._n_networks = 20
        else:
            if not (isinstance(n_networks, int) and n_networks > 0):
                raise ValueError(
                    "**NNetAR: error**: n_networks must be a positive integer or None"
                )
            else:
                self._n_networks = n_networks

        if auto and isinstance(self.period, int) and self.period in {1, 4, 7, 12}:
            self._deseasonalize = True
        else:
            self._deseasonalize = False
        self._deseasonalizer = None  # this may get reset later; just to avoid warnings
        self._scaler = None  # this may get reset later; just to avoid warnings
        self._ensemble = None  # this may get reset later; just to avoid warnings

        # self._deseasonalize = True ## force this for testing

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals should be used or fitted if needed
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

        def get_winlen_subset(p, P, period):
            if P == 0 or period is None:
                subset = None
                window_length = p
            else:
                subset = [i * period for i in range(P)] + list(
                    range(P * period - p, P * period)
                )
                window_length = max(p, P * period)
            return window_length, subset

        z = y.copy()

        if self._deseasonalize:
            self._deseasonalizer = Deseasonalizer(
                sp=self.period, model="multiplicative"
            )

            period_map = {12: "M", 4: "Q", 1: "A", 7: "W"}
            freq = period_map.get(self.period)

            if not isinstance(y.index, pd.PeriodIndex):
                z.index = z.index.to_period(freq)

            self._deseasonalizer.fit(z)
            z_deseas = self._deseasonalizer.transform(z)
            z_deseas.index = z.index.to_timestamp(freq)

            z = z_deseas.copy()

        if self.scale_inputs:
            self._scaler = StandardScaler()
            if isinstance(z, pd.Series):
                z = z.to_frame()
            z_scaled = pd.DataFrame(
                self._scaler.fit_transform(z), index=z.index, columns=z.columns
            )
            z = z_scaled.copy()

        regressor = MLPRegressor(
            hidden_layer_sizes=(self._n_nodes,),
            shuffle=False,
            activation="logistic",
            max_iter=2000,
            solver="lbfgs",
        )
        ##activation='relu',max_iter=2000, solver='lbfgs') ## 'logistic' is consistent
        # with R's NNETAR which calls 'nnet' which uses logistic

        if not self.auto:
            window_length, subset = get_winlen_subset(self._p, self._P, self.period)
            mlp_fc = RecursiveReductionForecaster(
                regressor, window_length=window_length, subset=subset
            )
            self._ensemble = EnsembleForecaster(
                forecasters=[("model", mlp_fc, self._n_networks)]
            )
            self._ensemble.fit(z)
            return

        # the auto case
        aic_values = []
        for P in range(0, self.max_P + 1):
            for p in range(1, self.max_p + 1):
                window_length, subset = get_winlen_subset(p, P, self.period)

                mlp_fc = RecursiveReductionForecaster(
                    regressor, window_length=window_length, subset=subset
                )
                ensemble = EnsembleForecaster(
                    forecasters=[("model", mlp_fc, self._n_networks)]
                )
                print(f"aic loop, p = {p}, P = {P}")
                ensemble.fit(z)

                fitted = ensemble.predict(z.index[window_length:])
                residuals = z.iloc[window_length:] - fitted
                n = len(residuals)
                k = p + P + 1
                aic = n * math.log((residuals**2).sum() / n) + 2 * k
                aic_values.append((aic, p, P))

        best_aic, best_p, best_P = min(aic_values, key=lambda x: x[0])

        print(f"best_aic = {best_aic}, best_p = {best_p}, best_P = {best_P}")

        mlp_fc = RecursiveReductionForecaster(
            regressor, window_length=window_length, subset=subset
        )
        self._ensemble = EnsembleForecaster(
            forecasters=[("model", mlp_fc, self._n_networks)]
        )
        self._ensemble.fit(z)
        return

    # todo: implement this, mandatory
    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        # implement here
        # IMPORTANT: avoid side effects to X, fh

        y_pred = self._ensemble.predict(fh)
        if not (self._deseasonalize or self.scale_inputs):
            return y_pred

        if self.scale_inputs:
            y_pred = pd.DataFrame(
                self._scaler.inverse_transform(y_pred),
                index=y_pred.index,
                columns=y_pred.columns,
            )

        if self._deseasonalize:
            if not isinstance(y_pred.index, pd.PeriodIndex):
                period_map = {12: "M", 4: "Q", 1: "A", 7: "W"}
                freq = period_map.get(self.period)
                y_pred.index = y_pred.index.to_period(freq)

            y_pred = self._deseasonalizer.inverse_transform(y_pred)
            y_pred.index = y_pred.index.to_timestamp(freq)

        return y_pred

    # todo: consider implementing this, optional
    # if not implementing, delete the _update method
    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """

        # implement here
        # IMPORTANT: avoid side effects to X, fh

    # todo: consider implementing this, optional
    # if not implementing, delete the _update_predict_single method
    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        self.update(y, X, update_params=update_params)
        return self.predict(fh, X)
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh

    # todo: consider implementing one of _predict_quantiles and _predict_interval
    #   if one is implemented, the other one works automatically
    #   when interfacing or implementing, consider which of the two is easier
    #   both can be implemented if desired, but usually that is not necessary
    #
    # if _predict_var or _predict_proba is implemented, this will have a default
    #   implementation which uses _predict_proba or _predict_var under normal assumption
    #
    # if implementing _predict_interval, delete _predict_quantiles
    # if not implementing either, delete both methods
    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh, alpha
        #
        # Note: unlike in predict_quantiles where alpha can be float or list of float
        #   alpha in _predict_quantiles is guaranteed to be a list of float

    # implement one of _predict_interval or _predict_quantiles (above), or delete both
    #
    # if implementing _predict_quantiles, delete _predict_interval
    # if not implementing either, delete both methods
    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh, coverage
        #
        # Note: unlike in predict_interval where coverage can be float or list of float
        #   coverage in _predict_interval is guaranteed to be a list of float

    # todo: consider implementing _predict_var
    #
    # if _predict_proba or interval/quantiles are implemented, this will have a default
    #   implementation which uses _predict_proba or quantiles under normal assumption
    #
    # if not implementing, delete _predict_var
    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        # implement here
        # implementing the cov=True case is optional and can be omitted

    # todo: consider implementing _predict_proba
    #
    # if interval/quantiles or _predict_var are implemented, this will have a default
    #   implementation which uses variance or quantiles under normal assumption
    #
    # if not implementing, delete _predict_proba
    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
            if has not been passed in fit, must be passed, not optional
        X : sktime time series object, optional (default=None)
                Exogeneous time series for the forecast
            Should be of same scitype (Series, Panel, or Hierarchical) as y in fit
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain fh.index and y.index both
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        # implement here
        # returned BaseDistribution should have same index and columns
        # as the predict return
        #
        # implementing the marginal=False case is optional and can be omitted

    # todo: consider implementing this, optional
    # if not implementing, delete the method
    def _predict_moving_cutoff(self, y, cv, X=None, update_params=True):
        """Make single-step or multi-step moving cutoff predictions.

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator
        X : pd.DataFrame
        update_params : bool

        Returns
        -------
        y_pred = pd.Series
        """

        # implement here
        # IMPORTANT: avoid side effects to y, X, cv

    # todo: consider implementing this, optional
    # implement only if different from default:
    #   default retrieves all self attributes ending in "_"
    #   and returns them with keys that have the "_" removed
    # if not implementing, delete the method
    #   avoid overriding get_fitted_params
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params
