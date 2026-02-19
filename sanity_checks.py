
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .config import ModelParams, PolicyName
from .model_common import unpack_state, shock_laws_of_motion
from .transforms import decode_outputs
from .residuals_a1 import residuals_a1
from .residuals_a2 import residuals_a2
from .residuals_a3 import residuals_a3


@dataclass
class FixedPointCheckResult:
    regime: int
    max_abs_state_diff: float


@dataclass
class ResidualCheckResult:
    regime: int
    max_abs_residual: float
    residuals: Dict[str, float]


def _state_from_policy_sss(params: ModelParams, policy: PolicyName, sss: Dict[str, float], regime: int) -> torch.Tensor:
    """Build a 1xN torch state vector x consistent with the project's state ordering."""
    dev, dt = params.device, params.dtype
    s = float(int(regime))

    if policy == "commitment":
        # x = (Delta_prev, logA, logg, xi, s, vp_prev, rp_prev)
        # where vp_prev = vartheta_prev * c_prev^gamma, rp_prev = varrho_prev * c_prev^gamma.
        # sss_from_policy stores vartheta_prev/varrho_prev already in that representation.
        vp_prev = float(sss.get("vartheta_prev", 0.0))
        rp_prev = float(sss.get("varrho_prev", 0.0))
        x = torch.tensor(
            [float(sss["Delta"]), float(sss["logA"]), float(sss["loggtilde"]), float(sss["xi"]), s, vp_prev, rp_prev],
            device=dev,
            dtype=dt,
        ).view(1, -1)
        return x

    # taylor, mod_taylor, discretion share x=(Delta_prev, logA, logg, xi, s)
    x = torch.tensor(
        [float(sss["Delta"]), float(sss["logA"]), float(sss["loggtilde"]), float(sss["xi"]), s],
        device=dev,
        dtype=dt,
    ).view(1, -1)
    return x


def _deterministic_next_state(
    params: ModelParams,
    policy: PolicyName,
    st,
    out: Dict[str, torch.Tensor],
    *,
    regime: int,
) -> torch.Tensor:
    """Compute x_{t+1} under zero innovations and fixed regime (as in the paper's SSS definition)."""
    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)

    if policy == "commitment":
        gamma = params.gamma
        vp_n = out["vartheta"] * out["c"].pow(gamma)
        rp_n = out["varrho"] * out["c"].pow(gamma)
        x_next = torch.stack(
            [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt), vp_n.view(-1), rp_n.view(-1)],
            dim=-1,
        )
        return x_next

    x_next = torch.stack(
        [out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)],
        dim=-1,
    )
    return x_next


def fixed_point_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, FixedPointCheckResult]:
    """
    Fixed point check at the SSS computed from the policy:
      - hold regime fixed
      - set innovations to zero
      - one-step deterministic transition should satisfy x_{t+1} ≈ x_t
    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    out_by_regime: Dict[int, FixedPointCheckResult] = {}
    for r, sss in sss_by_regime.items():
        x = _state_from_policy_sss(params, policy, sss, r)
        out = decode_outputs(policy, net(x), floors=floors)
        st = unpack_state(x, policy)
        x_next = _deterministic_next_state(params, policy, st, out, regime=r)
        out_by_regime[int(r)] = FixedPointCheckResult(regime=int(r), max_abs_state_diff=float((x_next - x).abs().max().item()))
    return out_by_regime


def _deterministic_terms_discretion(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.2 residuals.
    Returns (out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD).
    """
    x = x.clone().detach().requires_grad_(True)
    out = decode_outputs("discretion", net(x), floors=floors)
    st = unpack_state(x, "discretion")

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)

    def f_all():
        logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
        Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
        xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype)], dim=-1).view(1, -1)
        on = decode_outputs("discretion", net(xn), floors=floors)

        Lambda = params.beta * (on["lam"] / out["lam"])
        one_plus_pi = on["one_plus_pi"]

        # match deqn.py (Trainer._residuals): F, G, theta_term, XiN_rec, XiD_rec
        F = params.theta * params.beta * on["c"].pow(-params.gamma) * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
        G = params.theta * params.beta * on["c"].pow(-params.gamma) * one_plus_pi.pow(params.eps) * on["XiN"]
        theta_term = params.theta * one_plus_pi.pow(params.eps) * on["zeta"]
        XiN_rec = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
        XiD_rec = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]

        return F, G, theta_term, XiN_rec, XiD_rec

    Et_F, Et_G, Et_theta, Et_XiN, Et_XiD = f_all()

    Et_dF = torch.autograd.grad(Et_F.sum(), out["Delta"], create_graph=False, retain_graph=True)[0]
    Et_dG = torch.autograd.grad(Et_G.sum(), out["Delta"], create_graph=False, retain_graph=True)[0]

    return out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD


def _deterministic_terms_commitment(params: ModelParams, net: torch.nn.Module, x: torch.Tensor, *, regime: int, floors: Dict[str, float]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute deterministic (innovations=0, fixed regime) Et_* terms needed for Appendix A.3 residuals.
    Returns (out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta_zeta_pi).
    """
    x = x.clone().detach()
    out = decode_outputs("commitment", net(x), floors=floors)
    st = unpack_state(x, "commitment")

    dev, dt = params.device, params.dtype
    eps0 = torch.zeros(1, device=dev, dtype=dt)
    s_fixed = torch.full((1,), int(regime), device=dev, dtype=torch.long)
    gamma = params.gamma

    # build next state like in deqn.py
    logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
    Delta_cur = out["Delta"].view(-1, 1, 1).expand_as(logA_n)
    vp_cur = (out["vartheta"] * out["c"].pow(gamma)).view(-1, 1, 1).expand_as(logA_n)
    rp_cur = (out["varrho"] * out["c"].pow(gamma)).view(-1, 1, 1).expand_as(logA_n)
    xn = torch.stack([Delta_cur, logA_n, logg_n, xi_n, s_n.to(x.dtype), vp_cur, rp_cur], dim=-1).view(1, -1)

    on = decode_outputs("commitment", net(xn), floors=floors)

    Lambda = params.beta * (on["lam"] / out["lam"])
    one_plus_pi = on["one_plus_pi"]

    # match deqn.py (Trainer._residuals): XiN_rec, XiD_rec, termN, termD, theta_term
    Et_XiN = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
    Et_XiD = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]

    c_tg = out["c"].view(-1, 1, 1)
    termN = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(params.eps) * on["XiN"]
    termD = params.beta * params.theta * gamma * c_tg.pow(gamma - 1.0) * on["c"].pow(-gamma) * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
    theta_term = params.theta * one_plus_pi.pow(params.eps) * on["zeta"]

    return out, Et_XiN, Et_XiD, termN, termD, theta_term


def residuals_check(
    params: ModelParams,
    net: torch.nn.Module,
    *,
    policy: PolicyName,
    sss_by_regime: Dict[int, Dict[str, float]],
    floors: Optional[Dict[str, float]] = None,
) -> Dict[int, ResidualCheckResult]:
    """
    Residuals check evaluated at the SSS fixed point under:
      - innovations=0
      - fixed regime
    For discretion: 11 residuals (Appendix A.2).
    For commitment: 13 residuals (Appendix A.3).
    For Taylor variants: 8 residuals (Appendix A.1).
    """
    if floors is None:
        floors = {"c": 1e-8, "Delta": 1e-10, "pstar": 1e-10}

    results: Dict[int, ResidualCheckResult] = {}

    for r, sss in sss_by_regime.items():
        r = int(r)
        x = _state_from_policy_sss(params, policy, sss, r)

        if policy in ("taylor", "mod_taylor"):
            out = decode_outputs(policy, net(x), floors=floors)
            st = unpack_state(x, policy)

            # deterministic next (needed for Euler + Xi recursions)
            dev, dt = params.device, params.dtype
            eps0 = torch.zeros(1, device=dev, dtype=dt)
            s_fixed = torch.full((1,), r, device=dev, dtype=torch.long)

            logA_n, logg_n, xi_n, s_n = shock_laws_of_motion(params, st, eps0, eps0, eps0, s_fixed)
            xn = torch.stack([out["Delta"], logA_n.view(-1), logg_n.view(-1), xi_n.view(-1), s_n.to(dt)], dim=-1)

            on = decode_outputs(policy, net(xn), floors=floors)
            Lambda = params.beta * (on["lam"] / out["lam"])
            one_plus_pi = on["one_plus_pi"]
            Et_XiN = params.theta * Lambda * one_plus_pi.pow(params.eps) * on["XiN"]
            Et_XiD = params.theta * Lambda * one_plus_pi.pow(params.eps - 1.0) * on["XiD"]
            # Euler term uses the policy rule i_t. In some runs we don't store i in `out`
            # (it's a derived diagnostic), so compute it directly from the rule here.
            if "i" in out:
                i_t = out["i"]
            else:
                from .policy_rules import i_taylor, i_modified_taylor
                if policy == "taylor":
                    i_t = i_taylor(params, out["pi"])
                else:
                    # mod_taylor needs rbar_by_regime (natural-rate steady states)
                    from .steady_states import solve_flexprice_sss, export_rbar_tensor
                    rbar_by_regime = export_rbar_tensor(params, solve_flexprice_sss(params))
                    i_t = i_modified_taylor(params, out["pi"], rbar_by_regime, st.s)
            Et_eul = params.beta * ((1.0 + i_t) / one_plus_pi) * (on["lam"] / out["lam"])

            res = residuals_a1(params, st, out, Et_XiN, Et_XiD, Et_eul)

        elif policy == "discretion":
            out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD = _deterministic_terms_discretion(params, net, x, regime=r, floors=floors)
            st = unpack_state(x, "discretion")
            res = residuals_a2(params, st, out, Et_F, Et_G, Et_dF, Et_dG, Et_theta, Et_XiN, Et_XiD)

        elif policy == "commitment":
            out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta = _deterministic_terms_commitment(params, net, x, regime=r, floors=floors)
            st = unpack_state(x, "commitment")
            res = residuals_a3(params, st, out, Et_XiN, Et_XiD, Et_termN, Et_termD, Et_theta)

        else:
            raise ValueError(f"Unsupported policy for residual check: {policy}")

        # to python floats
        res_f = {k: float(v.detach().abs().max().item()) for k, v in res.items()}
        max_abs = max(res_f.values()) if res_f else float("nan")
        results[r] = ResidualCheckResult(regime=r, max_abs_residual=max_abs, residuals=res_f)

    return results
