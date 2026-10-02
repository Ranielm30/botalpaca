"""Check whether the notification path can actually reach Telegram."""

import asyncio

from botalpaca.app import Application
from botalpaca.config import get_settings
from botalpaca.db import Database


async def main():
    s = get_settings()
    db = Database(s.database_url, create_all=False, migrate=False)
    app = Application(s, database=db)
    await app.start()
    try:
        print(f"enabled={s.telegram_enabled} allowed={s.telegram_allowed_user_ids}")
        print("sender:", app.notifications._sender)
        try:
            await app.notifications._sender("PRUEBA DIRECTA botalpaca", None)
            print("SENDER OK")
        except Exception as exc:
            print(f"SENDER FALLO: {type(exc).__name__}: {exc}")
        print(f"chat ids configurados: {[c for c in dir(app) if 'chat' in c.lower()]}")
        try:
            await app.notifications.send("PRUEBA DE NOTIFICACION botalpaca", force=True)
            print("ENVIO OK")
        except Exception as exc:
            print(f"ENVIO FALLO: {type(exc).__name__}: {exc}")
        # And through the alert wrapper the monitors use.
        print(f"alert_handler definido: {app._alert_handler is not None}")
        try:
            await app.notify_position_alert(
                __import__("types").SimpleNamespace(
                    symbol="TEST",
                    environment=app.active_environment,
                    kind="fill",
                    position=None,
                    message="prueba",
                    changes=[],
                    fill_price=1.0,
                    stop_price=0.9,
                    target_price=1.2,
                    qty=1.0,
                    filled_at=None,
                    r_multiple=0.0,
                    detail="prueba",
                    previous_score=0.0,
                    current_score=0.0,
                )
            )
            print("ALERTA OK")
        except Exception as exc:
            print(f"ALERTA FALLO: {type(exc).__name__}: {exc}")
    finally:
        await app.stop()


asyncio.run(main())
