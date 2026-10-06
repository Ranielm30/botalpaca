"""Audit the LLY entry against what the market actually did.

Three questions the operator asked:
  1. Was it a good entry against the market?
  2. Were all its parameters right?
  3. Is the scanner assembling it correctly?

Answered from real data -- the recorded trade, the bars around it, and the
snapshot the scanner builds -- not from re-reading the code.
"""

import asyncio

from botalpaca.app import Application
from botalpaca.config import get_settings
from botalpaca.db import Database, TradeRepository
from botalpaca.domain import TradingEnvironment

PAPER = TradingEnvironment.PAPER
SYMBOL = "LLY"


async def main() -> None:
    s = get_settings()
    db = Database(s.database_url, create_all=False, migrate=False)
    app = Application(s, database=db)
    await app.start()
    try:
        # -- 1. What did we actually record? ---------------------------------
        print("=== 1. LA OPERACION REGISTRADA ===")
        async with db.session() as sess:
            rows = await TradeRepository(sess).get_all(PAPER, limit=200)
        trades = sorted(
            (t for t in rows if t.symbol == SYMBOL),
            key=lambda t: t.opened_at,
            reverse=True,
        )
        if not trades:
            print("  sin trades de LLY en la base")
        for t in trades[:3]:
            print(f"  id={t.id} abierto={t.opened_at} cerrado={t.closed_at}")
            print(f"     entrada={t.entry_price} stop={t.stop_price} tp={t.target_price}")
            print(f"     qty={t.qty} score={t.score} rr={t.rr} riesgo={t.risk_amount} pnl={t.pnl}")
            print(f"     estrategia={t.strategy} setup={t.setup} atr={t.atr}")
            print(f"     regimen={t.regime} sector={t.sector}")
            print(f"     motivo_entrada={t.entry_reason}")
            print(f"     motivo_salida={t.exit_reason_note}")

        # -- 2. What is Alpaca holding? --------------------------------------
        print("\n=== 2. ESTADO EN ALPACA ===")
        positions = [p for p in await app.active.portfolio.get_positions() if p.symbol == SYMBOL]
        if not positions:
            print("  LLY no tiene posicion abierta")
        for p in positions:
            print(f"  qty={p.qty} entrada={p.avg_entry_price} actual={p.current_price} "
                  f"pl={p.unrealized_pl} plpc={p.unrealized_plpc:.4f}")
            st = await app.active.protection.state_for_position(PAPER, p)
            print(f"  stop={st.has_stop}@{st.stop_price}  tp={st.has_take_profit}@{st.take_profit_price}")
            print(f"  trailing={st.has_trailing}  break_even={st.break_even_active}")
            print(f"  riesgo_entrada_congelado={st.initial_stop_price}")

        # -- 3. What did the market actually do? -----------------------------
        print("\n=== 3. QUE HIZO EL MERCADO (1h) ===")
        bars = await app.active.market.get_bars(SYMBOL, "1h", limit=200)
        if trades:
            since = trades[0].opened_at
            if since.tzinfo is not None:
                since = since.replace(tzinfo=None)
            # Bars carry the exchange timezone, the ledger stores UTC.
            import datetime as _dt

            def _utc(ts):
                if ts.tzinfo is None:
                    return ts.replace(tzinfo=_dt.UTC)
                return ts.astimezone(_dt.UTC).replace(tzinfo=None)

            after = [b for b in bars if _utc(b.timestamp) >= since]
            print(f"  entrada registrada: {since}")
            if after:
                lo = min(b.low for b in after)
                hi = max(b.high for b in after)
                last = after[-1]
                e = trades[0].entry_price
                print(f"  barras posteriores: {len(after)}  hasta {last.timestamp}")
                print(f"  min={lo:.2f} ({(lo / e - 1) * 100:+.2f}%)   "
                      f"max={hi:.2f} ({(hi / e - 1) * 100:+.2f}%)")
                print(f"  ultimo cierre={last.close:.2f} ({(last.close / e - 1) * 100:+.2f}%)")
                sp = trades[0].stop_price
                if sp:
                    hit = [b for b in after if b.low <= sp]
                    print(f"  stop en {sp:.2f}: {'TOCADO' if hit else 'nunca tocado'}"
                          f"   distancia inicial {(sp / e - 1) * 100:+.2f}%")
                tp = trades[0].target_price
                if tp:
                    hit = [b for b in after if b.high >= tp]
                    print(f"  target en {tp:.2f}: {'TOCADO' if hit else 'nunca tocado'}")
            else:
                print("  sin barras posteriores a la entrada")
        else:
            recent = bars[-30:]
            if recent:
                print(f"  min={min(b.low for b in recent):.2f} "
                      f"max={max(b.high for b in recent):.2f} "
                      f"ultimo={recent[-1].close:.2f}")

        # -- 4. Re-run the scanner NOW on the same symbol --------------------
        print("\n=== 4. LO QUE EL ESCANER VE HOY ===")
        outcome = await app.active.scanner.analyze_one(SYMBOL, timeframe="1h")
        if outcome is None:
            print("  el escaner no devolvio nada")
            return
        snap, opps = outcome
        ind = snap.indicators
        dump = ind.model_dump() if hasattr(ind, "model_dump") else dict(ind.__dict__)
        for name in sorted(dump):
            v = dump[name]
            if isinstance(v, (int, float)):
                print(f"    {name:18s} {v:,.4f}")
        st = getattr(snap, "structure", None)
        if st is not None:
            print(f"    soportes     {[f'{x.price:.2f}({x.touches}t)' for x in st.supports[:6]]}")
            print(f"    resistencias {[f'{x.price:.2f}({x.touches}t)' for x in st.resistances[:6]]}")
            print(f"    breakout={st.breakout} dir={st.breakout_direction}")
        print(f"  oportunidades: {len(opps)}")
        for o in opps[:3]:
            print(f"    {o.symbol} {o.direction.name} score={o.score} rr={o.rr} "
                  f"entry={o.entry} stop={o.stop} tp={o.target} tradable={o.tradable}")
            print(f"      motivo={o.non_tradable_reason}")
            print(f"      razones={o.reasons[:3]}")
    finally:
        await app.stop()


asyncio.run(main())
