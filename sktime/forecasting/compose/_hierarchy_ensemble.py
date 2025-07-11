# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecaster for applying different univariates on hierarchical data."""

__author__ = ["VyomkeshVyas"]
__all__ = ["HierarchyEnsembleForecaster"]


import pandas as pd

from sktime.base._meta import flatten
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.transformations.hierarchical.aggregate import _check_index_no_total
from sktime.utils.parallel import parallelize
from sktime.utils.warnings import warn


class HierarchyEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Aggregates hierarchical data, fit forecasters and make predictions.

    Can apply different univariate forecaster either on different
    level of aggregation or on different hierarchical nodes.

    ``HierarchyEnsembleForecaster`` is passed forecaster/level or
    forecaster/node pairs. Level can only be int >= 0 with 0
    signifying the topmost level of aggregation.
    Node can only be a tuple of strings or list of tuples.

    Behaviour in ``fit``, ``predict``:
    For level pairs ``f_i, l_i`` passed, applies forecaster ``f_i`` to level ``l_i``.
    For node pairs ``f_i, n_i`` passed,
    applies forecaster ``f_i`` on each node of ``n_i``.
    If ``default`` argument is passed, applies ``default`` forecaster on the
    remaining levels/nodes which are not mentioned in argument ``forecasters``.
    ``predict`` results are concatenated to one container with
    same columns as in ``fit``.

    Parameters
    ----------
    forecasters : sktime forecaster, or list of tuples
        (str, estimator, int or list of tuple/s)
        if forecaster, clones of ``forecaster`` are applied to all aggregated levels.
        if list of tuples, with name = str, estimator is forecaster, level/node
        as int/tuples respectively.
        all levels/nodes must be present in ``forecasters`` attribute if ``default``
        attribute is None

    by : {'node', 'level', default='level'}
        if ``'level'``, applies a univariate forecaster on all the hierarchical
        nodes within a level of aggregation
        if ``'node'``, applies separate univariate forecaster for each
        hierarchical node.

    default : sktime forecaster {default = None}
        if passed, applies ``default`` forecaster on the nodes/levels
        not mentioned in the ``forecaster`` argument.

    backend : string, by default "None".
        Parallelization backend to use for runs.
        Runs parallel evaluate if specified and ``strategy="refit"``.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".
    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings


    Examples
    --------
    >>> from sktime.forecasting.compose import HierarchyEnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=7,
    ...     no_levels=2,
    ...     random_seed=123,
    ... )

    >>> # Example of by = 'level'
    >>> forecasters = [
    ...     ('naive', NaiveForecaster(), 0),
    ...     ('trend', TrendForecaster(), 1),
    ... ]
    >>> forecaster = HierarchyEnsembleForecaster(
    ...     forecasters=forecasters,
    ...     by='level',
    ...     default=PolynomialTrendForecaster(degree=2),
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])
    HierarchyEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()

    >>> # Example of by = 'node'
    >>> forecasters = [
    ...     ('trend', TrendForecaster(), [("__total", "__total")]),
    ...     ('poly', PolynomialTrendForecaster(degree=2), [('l2_node01', 'l1_node01')]),
    ... ]
    >>> forecaster = HierarchyEnsembleForecaster(
    ...                 forecasters=forecasters,
    ...                 by='node', default=NaiveForecaster()
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])
    HierarchyEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _tags = {
        "authors": ["VyomkeshVyas", "sanskarmodi8"],
        "maintainers": ["VyomkeshVyas"],
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    BY_LIST = ["level", "node"]

    _steps_attr = "_forecasters"

    def __init__(
        self, forecasters, by="level", default=None, backend=None, backend_params=None
    ):
        self.forecasters = forecasters
        self.by = by
        self.default = default

        self.backend = backend
        self.backend_params = backend_params
        super().__init__(forecasters=None)

        if isinstance(forecasters, BaseForecaster):
            tags_to_clone = [
                "requires-fh-in-fit",
                "ignores-exogeneous-X",
                "capability:missing_values",
            ]
            self.clone_tags(forecasters, tags_to_clone)
        else:
            l_forecasters = [(x[0], x[1]) for x in forecasters]
            self._anytagis_then_set("requires-fh-in-fit", True, False, l_forecasters)
            self._anytagis_then_set("ignores-exogeneous-X", False, True, l_forecasters)
            self._anytagis_then_set(
                "capability:missing_values", False, True, l_forecasters
            )

    @property
    def _forecasters(self):
        """Make internal list of forecasters.

        The list only contains the name and forecasters. This is for the implementation
        of get_params via _HeterogenousMetaEstimator._get_params which expects lists of
        tuples of len 2.
        """
        forecasters = self.forecasters
        if isinstance(forecasters, BaseForecaster):
            return [("forecasters", forecasters)]
        else:
            return [(name, forecaster) for name, forecaster, _ in self.forecasters]

    @_forecasters.setter
    def _forecasters(self, value):
        if len(value) == 1 and isinstance(self.forecasters, BaseForecaster):
            self.forecasters = value[0][1]
        else:
            self.forecasters = [
                (name, forecaster, level_nd)
                for ((name, forecaster), (_, _, level_nd)) in zip(
                    value, self.forecasters
                )
            ]

    def _aggregate(self, y):
        """Add total levels to y, using Aggregate."""
        from sktime.transformations.hierarchical.aggregate import Aggregator

        return Aggregator().fit_transform(y)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd-multiindex
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        # Creating aggregated levels in data
        if _check_index_no_total(y):
            z = self._aggregate(y)
        else:
            z = y

        if X is not None:
            if _check_index_no_total(X):
                X = self._aggregate(X)

        x = X

        # check forecasters
        forecasters_ = self._check_forecasters(y, z)
        self.forecasters_ = self._ensure_clone(forecasters_)
        self.fitted_list_ = []

        if y.index.nlevels == 1:
            frcstr = self.forecasters_[0][1]
            frcstr.fit(y, fh=fh, X=X)
            self.fitted_list_.append([frcstr, y.index])
            return self

        meta = {"X": x, "z": z, "fh": fh}

        if self.by == "level":
            hier_dict = self._get_hier_dict(z)
            meta["hier_dict"] = hier_dict

            fcstr_list = parallelize(
                _level_fit,
                iter=self.forecasters_,
                meta=meta,
                backend=self.backend,
                backend_params=self.backend_params,
            )

            for i in range(len(fcstr_list)):
                self.fitted_list_.append(
                    [fcstr_list[i][0], fcstr_list[i][1].index.droplevel(-1).unique()]
                )
                name, _, level = self.forecasters_[i]
                self.forecasters_[i] = (name, fcstr_list[i][0], level)
        else:
            node_dict, frcstr_dict = self._get_node_dict(z)
            meta["fcstr_dict"] = frcstr_dict

            fcstr_list = parallelize(
                _node_fit,
                node_dict.items(),
                meta=meta,
                backend=self.backend,
                backend_params=self.backend_params,
            )

            for fcstr, df in fcstr_list:
                self.fitted_list_.append([fcstr, df.index.droplevel(-1).unique()])
        return self

    def _get_hier_dict(self, z):
        """Create a dictionary of hierarchy levels and MultiIndex object.

        Parameters
        ----------
        z : pd-multiindex
            Data to be segregated to hierarchical levels

        Returns
        -------
        hier_dict : dict
                    Dictionary with key as hierarchy level (int)
                    and values as MultiIndex
        """
        hier_dict = {}
        hier = z.index.droplevel(-1).unique()
        nlvls = z.index.nlevels

        _, _, levels = zip(*self.forecasters_)

        level_flat = flatten(levels)
        level_set = set(level_flat)

        for i in range(1, nlvls + 1):
            if nlvls - i in level_set:
                if i == 1:
                    level = hier[hier.get_level_values(-i) != "__total"]
                    hier_dict[nlvls - i] = level
                elif i == nlvls:
                    level = hier[hier.get_level_values(-i + 1) == "__total"]
                    hier_dict[nlvls - i] = level
                else:
                    level = hier[hier.get_level_values(-i) != "__total"]
                    level_cp = hier[hier.get_level_values(-i + 1) != "__total"]
                    diff = level.difference(level_cp)
                    if len(diff) != 0:
                        hier_dict[nlvls - i] = diff

        return hier_dict

    def _get_node_dict(self, z):
        """Create dictionaries of nodes and forecasters linked with common key value.

        Parameters
        ----------
        z : pd-multiindex
            Data to be segregated to hierarchical nodes

        Returns
        -------
        node_dict : dict
                    Dictionary with key as int and value as
                    Index/MultiIndex
        frcstr_dict : dict
                    Dictionary with key as int and value as
                    forecaster
        """
        node_dict = {}
        frcstr_dict = {}
        nodes = []
        counter = 0
        zindex = z.index.droplevel(-1).unique()

        for _, forecaster, node in self.forecasters_:
            if z.index.nlevels == 2:
                mi = pd.Index(node)
                if counter == 0:
                    nodes = mi
                else:
                    # For nlevels = 2, 'nodes' is pd.Index object (L286)
                    nodes = nodes.append(mi)
            else:
                node_l = []
                for i in range(len(node)):
                    if (
                        isinstance(node[i], tuple)
                        and len(node[i]) == z.index.nlevels - 1
                    ):
                        node_l.append(node[i])
                    elif isinstance(node[i], str):
                        for ind in zindex:
                            if ind[0] == node[i]:
                                node_l.append(ind)
                    else:
                        for ind in zindex:
                            if ind[: len(node[i])] == node[i]:
                                node_l.append(ind)

                mi = pd.MultiIndex.from_tuples(node_l, names=z.index.names[:-1])
                nodes += node_l
            frcstr_dict[counter] = forecaster
            node_dict[counter] = mi
            counter += 1

        diff_nodes = z.index.droplevel(-1).unique().difference(nodes)
        if self.default and len(diff_nodes) > 0:
            frcstr_dict[counter] = self.default.clone()
            node_dict[counter] = diff_nodes

        return node_dict, frcstr_dict

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.DataFrame
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self.
        """
        z = y
        if _check_index_no_total(y):
            z = self._aggregate(y)

        if X is not None:
            if _check_index_no_total(X):
                X = self._aggregate(X)
        x = X

        for forecaster, ind in self.fitted_list_:
            if z.index.nlevels == 1:
                forecaster.update(z, X=x, update_params=update_params)
            else:
                df = z[z.index.droplevel(-1).isin(ind)]
                if X is not None:
                    x = X.loc[df.index]
                forecaster.update(df, X=x, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None):
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
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        if X is not None:
            if _check_index_no_total(X):
                X = self._aggregate(X)
        x = X

        meta = {"x": x, "fh": fh}

        preds = parallelize(
            _predict_one_forecaster,
            self.fitted_list_,
            meta,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        preds = pd.concat(preds, axis=0)
        preds.sort_index(inplace=True)
        return preds

    def _check_forecasters(self, y, z):
        """Raise error if BY is not defined correctly."""
        if self.by not in self.BY_LIST:
            raise ValueError(f"""BY must be one of {self.BY_LIST}.""")

        if y.index.nlevels < 1:
            raise ValueError(
                "Data should have multiindex with levels greater than or equal to 1."
            )

        forecasters = self.forecasters

        # if a single estimator is passed, replicate across levels
        if isinstance(forecasters, BaseForecaster):
            if self.by == "level":
                lvlrange = range(y.index.nlevels)
                lvls = [str(lvl) for lvl in lvlrange]
                forecaster_list = [forecasters.clone() for _ in lvlrange]
                return list(zip(lvls, forecaster_list, lvlrange))
            else:
                if z.index.nlevels > 1:
                    node = z.index.droplevel(-1).unique().tolist()
                else:
                    node = z.index.tolist()
                name = "forecasters"
                return [(name, forecasters, node)]

        if (
            forecasters is None
            or len(forecasters) == 0
            or not isinstance(forecasters, list)
        ):
            raise ValueError(
                "Invalid 'forecasters' attribute, 'forecasters' should be either a "
                "Baseforecaster class or a list of (name, estimator, int/list) tuples."
            )

        if not isinstance(self.default, BaseForecaster) and self.default is not None:
            raise ValueError(
                "Invalid 'default' attribute, 'default' should be a BaseForecaster"
            )

        for i in range(len(forecasters)):
            if not isinstance(forecasters[i], tuple):
                raise ValueError(
                    "Invalid 'forecasters' attribute, 'forecasters' should "
                    "be either a BaseForecaster class or a list of tuples: "
                    " [(name, estimator, int/list)]."
                )
            if self.by == "node":
                if not isinstance(forecasters[i][2], list):
                    raise ValueError(
                        "Incorrect format of 'forecasters' attribute being passed."
                        "The 'Nodes' should be a list of tuple/tuples."
                    )

        _, forecasters_, level_nd = zip(*forecasters)

        for forecaster in forecasters_:
            if not isinstance(forecaster, BaseForecaster):
                raise ValueError(
                    f"The estimator {forecaster.__class__.__name__} should be a "
                    f"BaseForecaster class."
                )

        if y.index.nlevels == 1:
            return self.forecasters

        if self.by == "level":
            level_flat = flatten(level_nd)
            level_set = set(level_flat)
            not_in_z_idx = level_set.difference(range(z.index.nlevels))
            z_lvls_not_found = set(range(z.index.nlevels)).difference(level_set)
            zlvls_nf = [str(lvl) for lvl in z_lvls_not_found]

            if len(not_in_z_idx) > 0:
                raise ValueError(
                    f"Level identifier must be integers within "
                    f"the range of the total number of levels, "
                    f"but found level identifiers that are not: {list(not_in_z_idx)}"
                )

            if len(level_set) != len(level_flat):
                raise ValueError(
                    f"Only one estimator per level required. Found {len(level_flat)} "
                    f" level names in forecasters arg, required: {len(level_set)}"
                )
            if self.default is None and len(z_lvls_not_found) > 0:
                raise ValueError(
                    f"One estimator per level required. Following level/levels of "
                    f"data are missing estimator : {list(z_lvls_not_found)}"
                )

            if self.default:
                forecaster_list = [self.default.clone() for _ in z_lvls_not_found]
                return forecasters + list(
                    zip(zlvls_nf, forecaster_list, z_lvls_not_found)
                )
        else:
            nodes_t = []
            for nodes in level_nd:
                if len(nodes) == 0:
                    raise ValueError("Nodes cannot be empty.")
                if z.index.nlevels == 2:
                    nodes_ix = pd.Index(nodes)
                    nodes_t += nodes
                else:
                    nodes_l = []
                    for i in range(len(nodes)):
                        if (
                            isinstance(nodes[i], tuple)
                            and len(nodes[i]) > z.index.nlevels - 1
                        ):
                            raise ValueError(
                                "Ideally, length of individual node should be "
                                "equal to N-1 (where N is number of levels in "
                                "multi-index) and must not exceed N-1."
                            )
                        elif (
                            isinstance(nodes[i], tuple)
                            and len(nodes[i]) < z.index.nlevels - 1
                        ) or isinstance(nodes[i], str):
                            zindex = z.index.droplevel(-1).unique()
                            flag = 0
                            inds = []
                            if isinstance(nodes[i], tuple):
                                for ind in zindex:
                                    if ind[: len(nodes[i])] == nodes[i]:
                                        inds.append(ind)
                                        flag = 1
                            else:
                                for ind in zindex:
                                    if ind[0] == nodes[i]:
                                        inds.append(ind)
                                        flag = 1
                            if flag == 0:
                                raise ValueError(
                                    "Node value must lie within "
                                    "multi-index of aggregated data"
                                )
                            else:
                                nodes_l += inds
                                warn(
                                    f"Ideally, length of individual node "
                                    f"in HierarchyEnsembleForecaster should be "
                                    f"equal to N-1 (where N is number of levels in "
                                    f"multi-index) and must not exceed N-1. The "
                                    f"forecaster will now be fitted to the "
                                    f"following nodes : {list(inds)}",
                                    obj=self,
                                )
                        elif (
                            isinstance(nodes[i], tuple)
                            and len(nodes[i]) == z.index.nlevels - 1
                        ):
                            nodes_l.append(nodes[i])
                        else:
                            raise RuntimeError(
                                "Unreachable condition. Check the format of nodes "
                                "being passed."
                            )

                    nodes_ix = pd.MultiIndex.from_tuples(
                        nodes_l, names=z.index.names[:-1]
                    )
                    nodes_t += nodes_l
                nodes_m = z.index.droplevel(-1).unique()[
                    z.index.droplevel(-1).unique().isin(nodes_ix)
                ]
                nodes_nm = nodes_ix.difference(nodes_m)

                if len(nodes_nm) > 0:
                    raise ValueError(
                        f"Individual node value must be a tuple of "
                        f"index/indices within the multi-index of aggregated "
                        f"dataframe and must not include timepoint index. "
                        f"Following node/nodes are not present in"
                        f"the multi-index of aggregated data: {nodes_nm.to_list()}"
                    )
            nodes_set = set(nodes_t)
            if len(nodes_set) != len(nodes_t):
                raise ValueError(
                    f"Duplicate nodes found in 'forecasters' attribute. "
                    f"Only one estimator per node required. Found {len(nodes_t)} "
                    f"nodes , required: {len(nodes_set)}."
                )
            if z.index.nlevels == 2:
                nodes_tx = pd.Index(nodes_t)
            else:
                nodes_tx = pd.MultiIndex.from_tuples(nodes_t, names=z.index.names[:-1])

            z_nds_not_found = z.index.droplevel(-1).unique().difference(nodes_tx)
            if self.default is None and len(z_nds_not_found) > 0:
                raise ValueError(
                    f"One estimator per node required. Following nodes of "
                    f"data are missing estimator : {list(z_nds_not_found)}"
                )

        return forecasters

    def _ensure_clone(self, forecasters):
        """Ensure that forecasters are cloned.

        This should be done by _check_forecasters, but the function is hard to
        understand and debug. This function is a safety net to ensure that
        all forecasters are cloned.

        This function, and _check_forecasters, should be refactored to be more
        readable and maintainable.
        """
        forecasters_ = []
        for name, forecaster, level_nd in forecasters:
            forecasters_.append((name, forecaster.clone(), level_nd))
        return forecasters_

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        # imports
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster

        params1 = {
            "forecasters": [("ptf", PolynomialTrendForecaster(), 0)],
            "by": "level",
            "default": NaiveForecaster(),
        }
        params2 = {
            "forecasters": [("naive", NaiveForecaster(), [("__total")])],
            "by": "node",
            "default": PolynomialTrendForecaster(),
        }
        params3 = {"forecasters": NaiveForecaster()}

        return [params1, params2, params3]


def _level_fit(params, meta):
    """Fit a single forecaster.

    Called from _fit if by=level was set to allow for parallelization.

    Parameters
    ----------
    params: tuples
        (str, estimator, int or list of tuple/s)
        If called from the _fit function this will contain the self.forecasters_ list
    meta: dict
        Meta dictionary for parallelization. Needs to contain (at least)
        the following keys:
        z: pd.Series
            The aggregated target time series
        X: pd.DataFrame, None
            The aggregated input data
        hier_dict: dict
            The level dictionary as created by the get_hier_dict function
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
    """
    _, forecaster, level = params
    z = meta["z"]
    X = meta["X"]
    hier_dict = meta["hier_dict"]
    if level in hier_dict.keys():
        frcstr = forecaster
        df = z[z.index.droplevel(-1).isin(hier_dict[level])]
        if X is not None:
            X = X.loc[df.index]
        frcstr.fit(df, fh=meta["fh"], X=X)
    return frcstr, df


def _node_fit(params, meta):
    """Fit a single forecaster.

    Called from _fit if by=node was set to allow for parallelization.

    Parameters
    ----------
    params: tuples
        (str, estimator, int or list of tuple/s)
        If called from the _fit function this will contain the node_dict from the
        get_node_dict function
    meta: dict
        Meta dictionary for parallelization. Needs to contain (at least) the
        following keys:
        z: pd.Series
            The aggregated target time series
        X: pd.DataFrame, None
            The aggregated input data
        fcstr_dict: dict
            The forecaster dictionary as created by the get_node_dict function
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
    """
    key, node = params
    z = meta["z"]
    X = meta["X"]
    frcstr = meta["fcstr_dict"][key]
    df = z[z.index.droplevel(-1).isin(node)]
    if X is not None:
        X = X.loc[df.index]
    frcstr.fit(df, fh=meta["fh"], X=X)
    return frcstr, df


def _predict_one_forecaster(params, meta):
    """Predict a single forecaster.

    Called from _predict to allow for parallelization.

    Parameters
    ----------
    params: tuples
        (estimator, str)
        If called from the _predict function this will contain the self.fitted list
    meta: dict
        Meta dictionary for parallelization. Needs to contain (at least)
        the following keys:
        X: pd.DataFrame, None
            The aggregated input data
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
    """
    X = meta["x"]
    fh = meta["fh"]
    if X is not None and X.index.nlevels > 1:
        X = X[X.index.droplevel(-1).isin(params[1])]
    pred = params[0].predict(fh=fh, X=X)
    return pred
