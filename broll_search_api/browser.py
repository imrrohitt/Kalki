from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from broll_search_api.config import settings


CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class BrowserSession:
    def __init__(self, playwright: Any, browser: Browser, context: BrowserContext) -> None:
        self._playwright = playwright
        self.browser = browser
        self.context = context

    async def new_page(self) -> Page:
        return await self.context.new_page()

    async def close(self) -> None:
        await self.context.close()
        await self.browser.close()
        await self._playwright.stop()


@asynccontextmanager
async def launch_browser() -> AsyncIterator[BrowserSession]:
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=settings.playwright_headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context(
        user_agent=CHROME_UA,
        viewport={"width": 1440, "height": 900},
        locale="en-US",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    session = BrowserSession(playwright, browser, context)
    try:
        yield session
    finally:
        await session.close()


async def goto(page: Page, url: str) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=settings.search_timeout_ms)
    if settings.page_settle_ms > 0:
        await page.wait_for_timeout(settings.page_settle_ms)


async def dismiss_consent(page: Page) -> None:
    selectors = [
        "#L2AGLb",
        "#bnp_btn_accept",
        "button#onetrust-accept-btn-handler",
        "button[aria-label='Accept all']",
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
        "button:has-text('Accept')",
        "button:has-text('Reject extra cookies')",
    ]
    for selector in selectors:
        try:
            button = page.locator(selector).first
            if await button.count() and await button.is_visible():
                await button.click(timeout=1500)
                return
        except Exception:
            continue
