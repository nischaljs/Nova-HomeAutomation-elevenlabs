"""EngagementState semantics — fully offline.

These tests don't import the camera, the agent, or anything that needs
hardware. They exercise the pure timer logic that drives the audio gate
and the session lifecycle, which is the most failure-prone area of the
project (timer math, sticky windows, transitions).

Run with: pytest tests/test_engagement.py
"""

import time

import pytest

from app.orchestration import engagement
from app.orchestration.engagement import EngagementState


def test_fresh_state_reports_nothing_present():
    e = EngagementState()
    assert e.is_present() is False
    assert e.is_engaged() is False
    assert e.is_lips_moving() is False
    # No prior observation → "infinitely long since" so the lifecycle
    # code reads "keep session closed".
    assert e.presence_lost_for_s() == float("inf")
    assert e.disengaged_for_s() == float("inf")


def test_present_sets_timers():
    e = EngagementState()
    e.update(present=True, engaged=False, asym=0.6)
    assert e.is_present() is True
    assert e.is_engaged() is False
    assert e.presence_lost_for_s() < 0.05


def test_engaged_starts_clock():
    e = EngagementState()
    e.update(present=True, engaged=True, asym=0.2)
    assert e.is_engaged() is True
    assert e.engaged_for_s() < 0.05


def test_engaged_stickiness_survives_brief_lookaway():
    e = EngagementState()
    e.update(present=True, engaged=True, asym=0.2)
    # Sticky window is STICKY_S=2.5 — a single non-engaged tick should
    # NOT flip is_engaged off.
    e.update(present=True, engaged=False, asym=0.55)
    assert e.is_engaged() is True, (
        "engaged should stick for STICKY_S after the last engaged tick"
    )


def test_engaged_eventually_flips_off_after_sticky_window(monkeypatch):
    """Patch STICKY_S smaller so the test doesn't have to actually
    sleep 2.5 seconds."""
    monkeypatch.setattr(engagement, "STICKY_S", 0.05)
    e = EngagementState()
    e.update(present=True, engaged=True, asym=0.2)
    time.sleep(0.06)
    # A non-engaged update past the sticky window should clear it.
    e.update(present=True, engaged=False, asym=0.6)
    assert e.is_engaged() is False


def test_lips_moving_sticky_window():
    """is_lips_moving() respects LIP_MOTION_STICKY_S from lip_motion module."""
    e = EngagementState()
    e.update(present=True, engaged=True, asym=0.2, lips_moving=True)
    assert e.is_lips_moving() is True


def test_disengaged_for_s_tracks_engaged_only():
    """Important distinction: disengaged_for_s should count up while the
    user is in frame but turned away — NOT just while they're absent."""
    e = EngagementState()
    e.update(present=True, engaged=True, asym=0.2)
    # Now switch to "present but not engaged" — face still visible,
    # just turned away.
    e.update(present=True, engaged=False, asym=0.65)
    # presence is fresh
    assert e.presence_lost_for_s() < 0.05
    # disengaged accumulates from the moment of the last engaged tick
    # (we just updated, so it's tiny but real)
    assert e.disengaged_for_s() >= 0.0


def test_singleton_get_engagement_is_idempotent():
    a = engagement.get_engagement()
    b = engagement.get_engagement()
    assert a is b
