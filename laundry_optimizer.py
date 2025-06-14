"""
laundry_optimizer.py
====================

Optimal cost calculator for any combination of clothes,
based on the pricing/pack rules discussed in ChatGPT thread (June 2025).

Requires:  pulp  (pip install pulp)
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
from pulp import LpProblem, LpMinimize, LpInteger, LpVariable, lpSum, LpStatus
import json
import logging

CATALOG = {
    "packs_mistos": [
        {"tipo": "20",  "capacidade": 20,  "limite_camisas": 2, "preco": 10.0},
        {"tipo": "40",  "capacidade": 40,  "limite_camisas": 4, "preco": 20.0},
        {"tipo": "60",  "capacidade": 60,  "limite_camisas": 5, "preco": 30.0},
        {"tipo": "80",  "capacidade": 80,  "limite_camisas": 5, "preco": 37.5},
        {"tipo": "100", "capacidade": 100, "limite_camisas": 6, "preco": 45.0},
        {"tipo": "150", "capacidade": 150, "limite_camisas": 6, "preco": 65.0},
        {"tipo": "200", "capacidade": 200, "limite_camisas": 7, "preco": 85.0},
    ],
    "packs_camisas": [
        {"tipo": "10", "capacidade": 10, "preco": 7.5},
        {"tipo": "20", "capacidade": 20, "preco": 14.0},
        {"tipo": "50", "capacidade": 50, "preco": 37.5},
    ],
    "packs_lencois": [
        {"tipo": "10", "capacidade": 10, "preco": 9.5},
        {"tipo": "20", "capacidade": 20, "preco": 18.0},
    ],
    "avulso": {
        "peca_variada": 0.80,
        "camisa": 0.75,
        "vestido_simples": 7.0,
        "vestido_frisado": 12.5,
        "fato": 5.5,
        "casaco": 3.5,
        "toalha": 3.5,
        "lencol": 1.0,
    },
}

DELIVERY_FEES = {
    "montijo": 5.0,
    "lisboa": 0.0,
    "porto": 0.0,
    "default": 5.0,
}

class LaundryOptimizer:
    _SPECIALS = [
        "vestido_simples",
        "vestido_frisado",
        "fato",
        "casaco",
        "toalha",
    ]
    _DEFAULT_ITEM_KEYS = list(CATALOG["avulso"].keys())

    def __init__(self, catalog: dict = CATALOG,
                 fees: dict = DELIVERY_FEES,
                 logger: logging.Logger | None = None):
        self.catalog = catalog
        self.fees = fees
        self.log = logger or logging.getLogger(__name__)

    def optimize_order(
        self,
        items: Dict[str, int],
        delivery_location: str = "default",
        solver_name: str | None = None,
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        order = {k: int(items.get(k, 0)) for k in self._DEFAULT_ITEM_KEYS}
        bad_keys = [k for k in items if k not in order]
        if bad_keys:
            raise ValueError(f"Unknown item types: {bad_keys}")

        fee = self.fees.get(delivery_location.lower(), self.fees["default"])
        self.log.info("Order %s | delivery=%s (%.2f €)",
                      order, delivery_location, fee)

        specials_cost = sum(order[k] * self.catalog["avulso"][k]
                            for k in self._SPECIALS)

        qty_variada = order["peca_variada"]
        qty_camisas = order["camisa"]
        qty_lencois = order["lencol"]

        prob = LpProblem("Laundry_Cost_Min", LpMinimize)

        x = {p["tipo"]: LpVariable(f"x_misto_{p['tipo']}", 0, cat=LpInteger)
             for p in self.catalog["packs_mistos"]}
        y = {p["tipo"]: LpVariable(f"y_cam_{p['tipo']}", 0, cat=LpInteger)
             for p in self.catalog["packs_camisas"]}
        z = {p["tipo"]: LpVariable(f"z_len_{p['tipo']}", 0, cat=LpInteger)
             for p in self.catalog["packs_lencois"]}

        a_var = LpVariable("a_variada", 0, cat=LpInteger)
        a_cam = LpVariable("a_camisa", 0, cat=LpInteger)
        a_len = LpVariable("a_lencol", 0, cat=LpInteger)

        s = {p["tipo"]: LpVariable(f"s_cam_{p['tipo']}", 0, cat=LpInteger)
             for p in self.catalog["packs_mistos"]}

        cost_mistos  = lpSum(p["preco"] * x[p["tipo"]]
                             for p in self.catalog["packs_mistos"])
        cost_camisas = lpSum(p["preco"] * y[p["tipo"]]
                             for p in self.catalog["packs_camisas"])
        cost_lencois = lpSum(p["preco"] * z[p["tipo"]]
                             for p in self.catalog["packs_lencois"])
        cost_avulso  = (
            self.catalog["avulso"]["peca_variada"] * a_var +
            self.catalog["avulso"]["camisa"]        * a_cam +
            self.catalog["avulso"]["lencol"]        * a_len
        )
        prob += cost_mistos + cost_camisas + cost_lencois + cost_avulso

        for p in self.catalog["packs_mistos"]:
            prob += s[p["tipo"]] <= p["limite_camisas"] * x[p["tipo"]]

        prob += (
            lpSum(s.values()) +
            lpSum(p["capacidade"] * y[p["tipo"]]
                  for p in self.catalog["packs_camisas"]) +
            a_cam
            >= qty_camisas
        )

        prob += (
            lpSum(p["capacidade"] * x[p["tipo"]] - s[p["tipo"]]
                  for p in self.catalog["packs_mistos"]) +
            a_var
            >= qty_variada
        )

        prob += (
            lpSum(p["capacidade"] * z[p["tipo"]]
                  for p in self.catalog["packs_lencois"]) +
            a_len
            >= qty_lencois
        )

        status = prob.solve(solver_name)
        if LpStatus[status] != "Optimal":
            raise RuntimeError(f"Solver failed: {LpStatus[status]}")

        packs_mistos   = {k: int(v.value()) for k, v in x.items() if v.value()}
        packs_camisas  = {k: int(v.value()) for k, v in y.items() if v.value()}
        packs_lencois  = {k: int(v.value()) for k, v in z.items() if v.value()}
        avulso_counts  = {
            "peca_variada": int(a_var.value()),
            "camisa":       int(a_cam.value()),
            "lencol":       int(a_len.value()),
        }
        s_alloc        = {k: int(v.value()) for k, v in s.items() if v.value()}

        var_cost = value = prob.objective.value()
        total_cost = round(specials_cost + var_cost + fee, 2)

        breakdown = {
            "packs_mistos":   self._sort_dict(packs_mistos),
            "packs_camisas":  self._sort_dict(packs_camisas),
            "packs_lencois":  self._sort_dict(packs_lencois),
            "avulso":         avulso_counts,
            "camisas_nos_mistos": s_alloc,
            "custos": {
                "pecas_especiais": round(specials_cost, 2),
                "packs_mistos":    round(cost_mistos.value(), 2),
                "packs_camisas":   round(cost_camisas.value(), 2),
                "packs_lencois":   round(cost_lencois.value(), 2),
                "avulso":          round(cost_avulso.value(), 2),
                "entrega":         fee,
            },
        }

        raw_vars = {v.name: int(v.value()) for v in prob.variables()}

        return total_cost, breakdown, raw_vars

    @staticmethod
    def _sort_dict(d: Dict[str, int]) -> Dict[str, int]:
        return {k: d[k] for k in sorted(d, key=lambda x: int(x))}

def optimize_order(items: Dict[str, int],
                   delivery_location: str = "default"
                   ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    return LaundryOptimizer().optimize_order(items, delivery_location)
