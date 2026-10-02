"""Process entrypoint: wires the container, the Telegram bot and the scheduler.

Usage::

    python -m botalpaca            # run the bot + background workers
    python -m botalpaca --check    # validate config and Alpaca auth, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from botalpaca.app import Application
from botalpaca.config.logging import configure_logging, get_logger
from botalpaca.config.settings import Settings, get_settings
from botalpaca.domain.errors import BotalpacaError, ConfigurationError
from botalpaca.notifications import render_account
from botalpaca.scheduler import Scheduler
from botalpaca.telegram import TelegramFacade, build_telegram_application, register_handlers

log = get_logger(__name__)


def _banner(account: object, environment: object) -> str:
    return "\n".join(
        [
            "",
            "=" * 60,
            f" botalpaca — entorno activo: {environment.value}",
            f" {render_account(account)}",
            "=" * 60,
            "",
        ]
    )


async def _check(settings: Settings) -> int:
    """Validate configuration, credentials and database, then exit."""
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    app = Application(settings)
    try:
        account = await app.start()
    except ConfigurationError as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 2
    except BotalpacaError as exc:
        print(f"ERROR de Alpaca: {exc}", file=sys.stderr)
        return 3
    health = await app.health()
    print(_banner(account, app.active_environment))
    print(f"Base de datos: {'OK' if health.database_ok else 'FALLO'}")
    print(f"Kill switch: {'ACTIVO' if health.kill_switch else 'inactivo'}")
    for environment in ("PAPER", "REAL"):
        cfg = settings.alpaca(environment)  # type: ignore[arg-type]
        print(f"  {environment}: {'configurado' if cfg.is_configured else 'sin credenciales'}")
    await app.stop()
    return 0


async def _run(settings: Settings) -> int:
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    if not settings.telegram_enabled:
        print(
            "TELEGRAM_BOT_TOKEN no está configurado. Usa --check para validar "
            "el resto de la configuración.",
            file=sys.stderr,
        )
        return 4

    app = Application(settings)
    account = await app.start()
    print(_banner(account, app.active_environment))

    facade = TelegramFacade(app)

    bot_app = build_telegram_application(settings.telegram_bot_token.get_secret_value())
    register_handlers(bot_app, facade)
    bot_app.bot_data["application"] = app

    async def sender(text: str, keyboard: object = None) -> None:
        for user_id in sorted(app.allowlist.allowed):
            try:
                await bot_app.bot.send_message(
                    chat_id=user_id, text=text, parse_mode="HTML", reply_markup=keyboard
                )
            except Exception:  # noqa: BLE001 - one bad chat must not stop the rest
                log.exception("notifications.delivery_failed", user_id=user_id)

    app.set_sender(sender)
    app.set_alert_handler(facade.handle_alert)

    scheduler = Scheduler(app)

    # Recovery: reconcile SQLite against Alpaca before anything else runs, so a
    # restart never leaves a position unprotected.
    await scheduler.run_pending()

    await bot_app.initialize()
    # CRITICAL: python-telegram-bot >= 21 deliberately does NOT fetch updates in
    # `Application.start()`; the updater must be started explicitly (or via
    # `run_polling`). Without this line the process runs happily, connects to
    # Alpaca and serves nothing, because no one ever calls getUpdates.
    await bot_app.updater.start_polling(drop_pending_updates=False)
    await bot_app.start()
    await scheduler.start()

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        log.info("main.shutdown_requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, signal_name), _request_stop)
        except (NotImplementedError, AttributeError):  # Windows
            pass

    log.info(
        "main.running",
        environment=app.active_environment.value,
        monitor=scheduler.running,
        account=account.account_number or account.account_id,
    )
    try:
        await stop_event.wait()
    finally:
        await scheduler.stop()
        # Stop fetching first, then let the application drain its update queue.
        if bot_app.updater is not None:
            await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()
        await app.stop()
        log.info("main.stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="botalpaca", description="Telegram + Alpaca trading bot")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and credentials, then exit",
    )
    parser.add_argument(
        "--log-level", default=None, help="override LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.log_level:
        settings = settings.model_copy(update={"log_level": args.log_level.upper()})

    try:
        if args.check:
            return asyncio.run(_check(settings))
        return asyncio.run(_run(settings))
    except ConfigurationError as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
