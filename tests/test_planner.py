"""Unit tests for the Planner."""
from __future__ import annotations

from src.planner.planner import Planner, PlanStep


def test_plan_add_widget_to_cart() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("add widget to cart")
    assert steps[0].action == "goto"
    assert "store" in steps[0].target
    assert steps[1].action == "click"
    assert steps[1].target == "widget-001"


def test_plan_add_gadget_to_cart() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("add gadget to cart")
    assert steps[1].target == "gadget-002"


def test_plan_checkout() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("checkout")
    assert steps[0].action == "goto"
    assert any(s.action == "click" for s in steps)


def test_plan_open_cart() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("open cart")
    assert steps[0].action == "goto"
    assert "/cart" in steps[0].target


def test_plan_open_dashboard() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("open dashboard")
    assert steps[0].action == "goto"
    assert "/dashboard" in steps[0].target


def test_plan_unknown_command() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    steps = p.plan("do something weird")
    assert steps[0].action == "stop"


def test_is_goal_complete_cart() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    assert p.is_goal_complete("open cart", "http://127.0.0.1:8765/store/cart.html", "Cart") is True
    assert p.is_goal_complete("open cart", "http://127.0.0.1:8765/store/index.html", "Store") is False


def test_is_goal_complete_dashboard() -> None:
    p = Planner(store_url="http://127.0.0.1:8765/store/index.html",
                dashboard_url="http://127.0.0.1:8765/dashboard/index.html")
    assert p.is_goal_complete("open dashboard", "http://127.0.0.1:8765/dashboard/index.html", "Dashboard") is True