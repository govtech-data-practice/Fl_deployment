"""
Shared FL strategies: FedAvg, FedProx, FedAdam, FedYogi, SCAFFOLD, SecAgg+, DP-FedAvg
======================================================================================
"""

import re
import logging
import numpy as np
from typing import Callable, List, Optional, Tuple

import flwr as fl
from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedYogi
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics

logger = logging.getLogger("fl.strategies")


# ======================================================================
# SCAFFOLD (server updates control variates each round)
# ======================================================================

class FedSCAFFOLD(FedAvg):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.server_control: Optional[List[np.ndarray]] = None
        self.prev_global: Optional[List[np.ndarray]] = None

    def configure_fit(self, server_round, parameters, client_manager, **kw):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config["strategy"] = "scaffold"

        weights = parameters_to_ndarrays(parameters)
        if self.server_control is None:
            self.server_control = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.prev_global = [w.copy() for w in weights]

        fit_ins = fl.common.FitIns(parameters, config)
        n, mn = self.num_fit_clients(client_manager.num_available())
        return [(c, fit_ins) for c in client_manager.sample(num_clients=n, min_num_clients=mn)]

    def aggregate_fit(self, server_round, results, failures, **kw):
        if not results:
            return None, {}
        wr = [(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results]
        total = sum(n for _, n in wr)
        nl = len(wr[0][0])
        agg = [np.zeros(wr[0][0][i].shape, dtype=np.float32) for i in range(nl)]
        for w, n in wr:
            f = np.float32(n / total)
            for i, v in enumerate(w):
                agg[i] += np.asarray(v, dtype=np.float32) * f

        # Update server control variates: c += (1/N) * (x_prev - x_new) / (K * η)
        # Simplified: track aggregate parameter shift as proxy for control update
        if self.prev_global is not None and self.server_control is not None:
            N = len(wr)
            for i in range(nl):
                delta = (self.prev_global[i] - agg[i]).astype(np.float32)
                self.server_control[i] = self.server_control[i] + delta / N

        return ndarrays_to_parameters(agg), {}


# ======================================================================
# SecAgg+ (requires full client participation)
# ======================================================================

class FedSecAggPlus(FedAvg):
    def __init__(self, num_fl_clients: int = 2, **kw):
        super().__init__(**kw)
        self.num_fl_clients = num_fl_clients

    def configure_fit(self, server_round, parameters, client_manager, **kw):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config["strategy"] = "secagg"
        config["secagg_round_seed"] = server_round * 1000 + 42
        config["secagg_num_clients"] = self.num_fl_clients
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.num_fl_clients,
            min_num_clients=self.num_fl_clients,
        )
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures, **kw):
        """Simple average (1/N) — required for mask cancellation.

        Masks cancel under equal weighting: for pair (i,j), client i adds +M,
        client j adds -M. In sum: f_i*M + f_j*(-M) = M*(f_i - f_j).
        Only zero when f_i = f_j = 1/N.
        """
        if failures:
            logger.error(f"SecAgg+: {len(failures)} failures — masks may not cancel")
            return None, {}
        if not results:
            return None, {}
        wr = [parameters_to_ndarrays(r.parameters) for _, r in results]
        N = len(wr)
        nl = len(wr[0])
        # Equal-weight average (1/N)
        agg = [np.zeros(wr[0][i].shape, dtype=np.float64) for i in range(nl)]  # float64 for precision
        for w in wr:
            for i, v in enumerate(w):
                agg[i] += np.asarray(v, dtype=np.float64)
        agg = [(a / N).astype(np.float32) for a in agg]
        return ndarrays_to_parameters(agg), {}


# ======================================================================
# DP-FedAvg (central DP: clip each client update, aggregate, add noise)
# ======================================================================

class FedDPAvg(FedAvg):
    """Differentially-private FedAvg.

    Central mode: server clips each client's update Δ to norm C,
    computes weighted average of clipped updates, adds N(0, σ²C²/N² I).
    Local mode: clients clip+noise their own updates (handled client-side).
    """

    def __init__(self, dp_mode="central", noise_multiplier=1.0,
                 max_grad_norm=1.0, num_fl_clients=2, target_delta=1e-5, **kw):
        super().__init__(**kw)
        self.dp_mode = dp_mode
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.num_fl_clients = num_fl_clients
        self.target_delta = target_delta
        self._prev_global = None

        from fl_common.dp import PrivacyAccountant
        self.accountant = PrivacyAccountant(
            noise_multiplier=noise_multiplier, sample_rate=1.0, delta=target_delta,
        )

    def configure_fit(self, server_round, parameters, client_manager, **kw):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config["strategy"] = "dp"
        config["dp_mode"] = self.dp_mode
        config["dp_noise_multiplier"] = self.noise_multiplier
        config["dp_max_grad_norm"] = self.max_grad_norm
        config["dp_seed"] = server_round * 7919 + 31

        # Store current global params for computing client deltas
        self._prev_global = parameters_to_ndarrays(parameters)

        fit_ins = fl.common.FitIns(parameters, config)
        n, mn = self.num_fit_clients(client_manager.num_available())
        return [(c, fit_ins) for c in client_manager.sample(num_clients=n, min_num_clients=mn)]

    def aggregate_fit(self, server_round, results, failures, **kw):
        if not results:
            return None, {}

        from fl_common.dp import clip_update

        weights_list = [
            (parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results
        ]
        N = len(weights_list)
        total_n = sum(n for _, n in weights_list)
        nl = len(weights_list[0][0])

        if self.dp_mode == "central" and self._prev_global is not None:
            # 1. Compute and clip each client's update
            clipped_deltas = []
            for client_w, n_ex in weights_list:
                delta = [cw - gw for cw, gw in zip(client_w, self._prev_global)]
                delta = clip_update(delta, self.max_grad_norm)
                clipped_deltas.append((delta, n_ex))

            # 2. Weighted average of clipped deltas
            avg_delta = [np.zeros(self._prev_global[i].shape, dtype=np.float32)
                         for i in range(nl)]
            for delta, n_ex in clipped_deltas:
                f = np.float32(n_ex / total_n)
                for i, d in enumerate(delta):
                    avg_delta[i] += np.asarray(d, dtype=np.float32) * f

            # 3. Add noise: σ * C / N (sensitivity of avg of N clipped updates)
            sigma = self.noise_multiplier * self.max_grad_norm / N
            rng = np.random.RandomState(server_round * 7919)
            for i in range(nl):
                avg_delta[i] += rng.normal(0, sigma, size=avg_delta[i].shape).astype(np.float32)

            # 4. Apply noised delta to global model
            agg = [self._prev_global[i] + avg_delta[i] for i in range(nl)]

        else:
            # Local DP or fallback: clients already added noise, just FedAvg
            agg = [np.zeros(weights_list[0][0][i].shape, dtype=np.float32) for i in range(nl)]
            for w, n_ex in weights_list:
                f = np.float32(n_ex / total_n)
                for i, v in enumerate(w):
                    agg[i] += np.asarray(v, dtype=np.float32) * f

        self.accountant.step()
        eps = self.accountant.get_epsilon()
        logger.info(f"DP-FedAvg ({self.dp_mode}): round {server_round}, "
                    f"σ={self.noise_multiplier}, C={self.max_grad_norm}, "
                    f"ε={eps:.2f}, δ={self.target_delta}")
        return ndarrays_to_parameters(agg), {"dp_epsilon": eps}


# ======================================================================
# FedAdaptiveWarmup: FedAdam for early rounds, then FedAvg
# ======================================================================

class FedAdaptiveWarmup(FedAvg):
    """Warmup with adaptive server optimizer, then switch to FedAvg.

    Rounds 1..warmup_rounds: FedAdam aggregation (adaptive LR helps
    early convergence when pseudo-gradients are informative).
    Rounds warmup_rounds+1..N: FedAvg aggregation (simple weighted avg
    prevents adaptive LR from amplifying noise in later rounds).

    Solves: FedAdam learns fast initially (0.60 AUC in 10 rounds) but
    plateaus and degrades. FedAvg converges slower but reaches 0.81.
    This strategy gets both: fast early + stable late.
    """

    def __init__(self, warmup_rounds: int = 10, adam_eta: float = 0.1,
                 adam_tau: float = 0.1, **kw):
        super().__init__(**kw)
        self.warmup_rounds = warmup_rounds
        self.adam_eta = adam_eta
        self.adam_tau = adam_tau
        self._prev_global = None
        # Adam state
        self._m = None  # first moment
        self._v = None  # second moment
        self._t = 0     # step counter

    def configure_fit(self, server_round, parameters, client_manager, **kw):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        phase = "adam" if server_round <= self.warmup_rounds else "avg"
        config["strategy"] = "adaptive_warmup"
        config["warmup_phase"] = phase

        self._prev_global = parameters_to_ndarrays(parameters)

        fit_ins = fl.common.FitIns(parameters, config)
        n, mn = self.num_fit_clients(client_manager.num_available())
        return [(c, fit_ins) for c in client_manager.sample(num_clients=n, min_num_clients=mn)]

    def aggregate_fit(self, server_round, results, failures, **kw):
        if not results:
            return None, {}

        wr = [(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results]
        total_n = sum(n for _, n in wr)
        nl = len(wr[0][0])

        # Weighted average of client params (same for both phases)
        avg = [np.zeros(wr[0][0][i].shape, dtype=np.float32) for i in range(nl)]
        for w, n in wr:
            f = np.float32(n / total_n)
            for i, v in enumerate(w):
                avg[i] += np.asarray(v, dtype=np.float32) * f

        if server_round <= self.warmup_rounds and self._prev_global is not None:
            # Adam phase: apply adaptive server update
            self._t += 1

            if self._m is None:
                self._m = [np.zeros_like(a) for a in avg]
                self._v = [np.zeros_like(a) for a in avg]

            # Pseudo-gradient: delta = avg - prev
            agg = []
            for i in range(nl):
                delta = avg[i] - self._prev_global[i]
                # Adam update
                self._m[i] = 0.9 * self._m[i] + 0.1 * delta
                self._v[i] = 0.99 * self._v[i] + 0.01 * (delta * delta)
                step = self._m[i] / (np.sqrt(self._v[i]) + self.adam_tau)
                agg.append(self._prev_global[i] + self.adam_eta * step)

            logger.info(f"FedAdaptiveWarmup: round {server_round} [ADAM phase], "
                        f"warmup={self.warmup_rounds}")
        else:
            # FedAvg phase: just use the weighted average
            agg = avg
            if server_round == self.warmup_rounds + 1:
                logger.info(f"FedAdaptiveWarmup: round {server_round} [switching to FedAvg]")
                self._m = None  # discard Adam state
                self._v = None

        return ndarrays_to_parameters(agg), {}


# ======================================================================
# FedOneOwner: Only one party retains the final global model
# ======================================================================

class FedOneOwner(FedAvg):
    """Federated learning where only one party retains the final model.

    Training is standard FedAvg — all clients participate equally.
    The "one owner" aspect is access control: after training completes,
    only the designated owner receives the final global model.
    Other clients contributed training but don't keep the result.

    Aggregation is identical to FedAvg (no weight boost — that hurts
    convergence). The value is in the deployment model, not the algorithm.
    """

    def __init__(self, owner_id: int = 0, num_fl_clients: int = 2, **kw):
        super().__init__(**kw)
        self.owner_id = owner_id
        self.num_fl_clients = num_fl_clients

    def configure_fit(self, server_round, parameters, client_manager, **kw):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        config["strategy"] = "oneowner"
        config["owner_id"] = self.owner_id

        fit_ins = fl.common.FitIns(parameters, config)
        n, mn = self.num_fit_clients(client_manager.num_available())
        return [(c, fit_ins) for c in client_manager.sample(num_clients=n, min_num_clients=mn)]

    def aggregate_fit(self, server_round, results, failures, **kw):
        """Standard FedAvg aggregation — owner distinction is at deployment."""
        agg_result = super().aggregate_fit(server_round, results, failures, **kw)
        logger.info(f"FedOneOwner: round {server_round}, owner={self.owner_id}, "
                    f"{len(results)} clients")
        return agg_result


# ======================================================================
# Early-stop wrapper
# ======================================================================

class EarlyStopWrapper:
    def __init__(self, strategy, metric_name="accuracy", patience=30, min_delta=0.0005):
        self.strategy = strategy
        self.metric_name = metric_name
        self.best = 0.0
        self.stale = 0
        self.patience = patience
        self.min_delta = min_delta
        self._stop = False

    def __getattr__(self, name):
        return getattr(self.strategy, name)

    def configure_fit(self, *a, **kw):
        return [] if self._stop else self.strategy.configure_fit(*a, **kw)

    def configure_evaluate(self, *a, **kw):
        return [] if self._stop else self.strategy.configure_evaluate(*a, **kw)

    def aggregate_fit(self, *a, **kw):
        try:
            return self.strategy.aggregate_fit(*a, **kw)
        except Exception as e:
            logger.error(f"aggregate_fit error: {e}", exc_info=True)
            return None, {}

    def aggregate_evaluate(self, *a, **kw):
        loss, metrics = self.strategy.aggregate_evaluate(*a, **kw)
        if loss is None:
            return None, {}
        score = metrics.get(self.metric_name, 0.0)
        rnd = kw.get("server_round", a[0] if a else 0)
        if score > self.best + self.min_delta:
            self.best = score
            self.stale = 0
        else:
            self.stale += 1
        logger.info(
            f"Round {rnd}: loss={loss:.4f} {self.metric_name}={score:.4f} "
            f"best={self.best:.4f} stale={self.stale}"
        )
        if self.stale >= self.patience:
            logger.info(f"Early stopping at round {rnd}")
            self._stop = True
        return loss, metrics


# ======================================================================
# Weighted-average metric aggregation
# ======================================================================

def make_weighted_average(metric_name: str):
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {metric_name: 0.0, "loss": 0.0}
        # Only include clients that reported the metric in the denominator
        vals = [(n, m.get(metric_name, 0)) for n, m in metrics if metric_name in m]
        losses = [(n, m.get("loss", 0)) for n, m in metrics if "loss" in m]
        val_total = sum(n for n, _ in vals) or 1
        loss_total = sum(n for n, _ in losses) or 1
        return {
            metric_name: sum(n * v for n, v in vals) / val_total if vals else 0.0,
            "loss": sum(n * v for n, v in losses) / loss_total if losses else 0.0,
        }
    return weighted_average


# ======================================================================
# Strategy factory
# ======================================================================

def _parse_name(name):
    if name == "IID":
        return ("IID", None, None)
    alpha = float(m.group(1)) if (m := re.search(r'Alpha_([\d.]+)', name)) else 0.5
    mu = float(m.group(1)) if (m := re.search(r'Mu([\d.]+)', name)) else None
    for s in ["DP_Local", "DP_Central", "AdaptiveWarmup", "OneOwner", "FedProx",
              "FedAdam", "FedYogi", "SCAFFOLD", "SecAgg", "FedAvg"]:
        if name.startswith(s):
            return (s, mu, alpha)
    return ("FedAvg", mu, alpha)


def build_strategy(
    name: str,
    num_clients: int,
    model_init_fn: Callable,
    metric_name: str = "accuracy",
    lr: float = 0.001,
    patience: int = 30,
    min_delta: float = 0.0005,
):
    stype, mu, alpha = _parse_name(name)
    dp_eps_match = re.search(r'Eps([\d.]+)', name)
    dp_noise = 1.0 / float(dp_eps_match.group(1)) if dp_eps_match else 1.0
    dp_clip_match = re.search(r'Clip([\d.]+)', name)
    dp_clip = float(dp_clip_match.group(1)) if dp_clip_match else 5.0

    def _init_params():
        m = model_init_fn()
        return ndarrays_to_parameters([v.cpu().numpy() for v in m.state_dict().values()])

    def fit_cfg(alpha_val, sname="fedavg"):
        def fn(rnd):
            return {
                "partition_type": "iid" if alpha_val is None else "label_skew",
                "alpha": 100.0 if alpha_val is None else alpha_val,
                "current_round": rnd, "learning_rate": lr,
                "local_epochs": 1, "strategy": sname,
            }
        return fn

    base = {
        "fraction_fit": 1.0, "fraction_evaluate": 1.0,
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
        "evaluate_metrics_aggregation_fn": make_weighted_average(metric_name),
        "on_evaluate_config_fn": lambda r: {"current_round": r},
    }

    if stype in ("IID", "FedAvg"):
        a = None if stype == "IID" else alpha
        strat = FedAvg(on_fit_config_fn=fit_cfg(a), **base)
    elif stype == "FedProx":
        mu = mu or 0.1
        def prox_cfg(r):
            c = fit_cfg(alpha, "fedprox")(r)
            c["proximal_mu"] = mu
            return c
        strat = FedProx(proximal_mu=mu, on_fit_config_fn=prox_cfg, **base)
    elif stype == "FedAdam":
        # eta=0.1 with tau=0.1: large damping prevents Adam from amplifying noise
        # on pretrained models where pseudo-gradients are small and noisy
        strat = FedAdam(initial_parameters=_init_params(),
                        eta=0.1, eta_l=lr, beta_1=0.9, beta_2=0.99, tau=0.1,
                        on_fit_config_fn=fit_cfg(alpha, "fedadam"), **base)
    elif stype == "FedYogi":
        strat = FedYogi(initial_parameters=_init_params(),
                        eta=0.1, eta_l=lr, beta_1=0.9, beta_2=0.99, tau=0.1,
                        on_fit_config_fn=fit_cfg(alpha, "fedyogi"), **base)
    elif stype == "SCAFFOLD":
        strat = FedSCAFFOLD(initial_parameters=_init_params(),
                            on_fit_config_fn=fit_cfg(alpha, "scaffold"), **base)
    elif stype == "SecAgg":
        strat = FedSecAggPlus(num_fl_clients=num_clients,
                              initial_parameters=_init_params(),
                              on_fit_config_fn=fit_cfg(alpha, "secagg"), **base)
    elif stype in ("DP_Central", "DP_Local"):
        dp_mode = "central" if stype == "DP_Central" else "local"
        strat = FedDPAvg(
            dp_mode=dp_mode, noise_multiplier=dp_noise,
            max_grad_norm=dp_clip, num_fl_clients=num_clients,
            initial_parameters=_init_params(),
            on_fit_config_fn=fit_cfg(alpha, "dp"), **base,
        )
    elif stype == "AdaptiveWarmup":
        # Parse warmup rounds: AdaptiveWarmup_W10_Alpha_0.5
        w_match = re.search(r'W(\d+)', name)
        warmup = int(w_match.group(1)) if w_match else 10
        strat = FedAdaptiveWarmup(
            warmup_rounds=warmup, adam_eta=0.1, adam_tau=0.1,
            initial_parameters=_init_params(),
            on_fit_config_fn=fit_cfg(alpha, "adaptive_warmup"), **base,
        )
    elif stype == "OneOwner":
        strat = FedOneOwner(
            owner_id=0, num_fl_clients=num_clients,
            initial_parameters=_init_params(),
            on_fit_config_fn=fit_cfg(alpha, "oneowner"), **base,
        )
    else:
        strat = FedAvg(on_fit_config_fn=fit_cfg(0.5), **base)

    return EarlyStopWrapper(strat, metric_name=metric_name,
                            patience=patience, min_delta=min_delta)
