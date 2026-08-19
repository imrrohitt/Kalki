import pytest


@pytest.fixture(autouse=True)
def disable_editorial_llm(monkeypatch):
    monkeypatch.setattr("app.config.settings.editorial_llm_enabled", False)
    monkeypatch.setattr("app.editorial.analyzer.settings.editorial_llm_enabled", False)
    monkeypatch.setattr("app.config.settings.graphics_llm_enabled", False)
    monkeypatch.setattr("app.config.settings.sfx_llm_enabled", False)
