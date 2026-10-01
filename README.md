# botalpaca

Asistente profesional de trading por Telegram integrado con **Alpaca**, con entornos
**PAPER** y **REAL** totalmente aislados.

Todo el ciclo de trabajo ocurre desde Telegram: análisis, scoring, dimensionamiento,
ejecución, protección de la posición, seguimiento y estadísticas. No hace falta entrar
al panel de Alpaca para operar.

> **Este software no garantiza ganancias.** Ninguna señal es una recomendación
> financiera. Opera en PAPER hasta que entiendas su comportamiento.

---

## 1. Arquitectura (19 capas)

| # | Capa | Módulo | Responsabilidad |
|---|------|--------|-----------------|
| 1 | Interfaz Telegram | `botalpaca/telegram/` | Comandos, teclado inline, callbacks, parseo de argumentos |
| 2 | Aplicación / servicios | `botalpaca/app.py` | Contenedor de inyección de dependencias, ciclo de vida, cambio de entorno |
| 3 | Dominio de trading | `botalpaca/domain/` | Enums, errores y modelos Pydantic (`extra="forbid"`) |
| 4 | Motor de estrategias | `botalpaca/strategies/` | 12 estrategias independientes y combinables |
| 5 | Scanner de mercado | `botalpaca/scanner/` | Recorre el universo, deduplica por fingerprint, prioriza |
| 6 | Análisis técnico | `botalpaca/indicators/`, `botalpaca/analysis/` | Indicadores puros, tendencia, momentum, volumen, volatilidad, estructura |
| 7 | Gestión de riesgo | `botalpaca/risk/` | Tamaño de posición y 18 reglas de bloqueo/trim |
| 8 | Gestión de posiciones | `botalpaca/protection/` | Stops, break-even, trailing seguro, progresivo, time stop, reconciliación |
| 9 | Ejecución / adaptador Alpaca | `botalpaca/execution/` | Constructor de órdenes respetando las restricciones reales de Alpaca, barrera de entorno, idempotencia |
| 10 | Portfolio / cuenta | `botalpaca/portfolio/` | Lecturas de cuenta, posiciones, órdenes, exposición |
| 11 | Trade journal | `botalpaca/journal/` | Cada señal y cada operación, con el motivo de entrada y de salida |
| 12 | Aprendizaje estadístico | `botalpaca/learning/` | Win rate, profit factor, expectancy, Sharpe (con muestra suficiente), MFE/MAE |
| 13 | Scoring de oportunidad | `botalpaca/confluence/` | Score 0-100, calidad ALTA/MEDIA/BAJA/NO OPERABLE, explicación de entrada |
| 14 | Monitoring | `botalpaca/monitoring/` | Monitor de mercado y monitor de posiciones en horario de sesión |
| 15 | Notificaciones | `botalpaca/notifications/` | Renderizadores de tarjetas y presupuesto de alertas |
| 16 | Persistencia | `botalpaca/db/` | SQLAlchemy async + Alembic, 8 tablas, repositorios que exigen `environment` |
| 17 | Configuración | `botalpaca/config/` | Settings tipados, logging estructurado con redacción de secretos |
| 18 | Seguridad | `botalpaca/security/` | Allowlist, rate limit, circuit breaker, kill switch, confirmaciones |
| 19 | Scheduler | `botalpaca/scheduler/` | Bucles de scan, monitor y reconciliación |

Orden de prioridad del sistema:
**SEGURIDAD > CORRECTA EJECUCIÓN > GESTIÓN DE RIESGO > CALIDAD DE DATOS > ESTADÍSTICAS > ESTRATEGIAS > VELOCIDAD**

---

## 2. Aislamiento PAPER / REAL

Dos entornos **completamente independientes**:

- Claves API y base URL propias (`ALPACA_PAPER_*` / `ALPACA_LIVE_*`).
- Cliente `TradingClient` propio por entorno.
- Una sola cuenta Alpaca activa a la vez: `ACTIVE_TRADING_ENVIRONMENT = PAPER | REAL`.

Garantías implementadas:

1. `ExecutionEngine` **rechaza** cualquier orden cuyo `environment` no coincida con el
   entorno activo (`EnvironmentMismatchError`). El cliente se inyecta ya ligado a su entorno,
   así que el constructor falla si se mezclan.
2. El motor de datos de mercado (scanner) es compartido; el estado financiero **nunca** lo es.
3. `/posiciones`, `/ordenes`, `/cuenta`, `/portfolio` muestran **solo** el entorno activo.
4. `/historial`, `/stats` y el Trade Journal están particionados por entorno. **Cada consulta a
   la base de datos exige un `environment`**; no existe un método "traer todo".
5. PAPER es el valor por defecto al arrancar. Un entorno REAL persistido **no** sobrescribe
   el entorno configurado tras un reinicio.
6. Nunca hay auto-cambio de PAPER a REAL. `/modo` exige confirmación explícita, verifica la
   autenticación, consulta la cuenta y **no envía ninguna orden** durante el cambio.
7. Toda operación REAL muestra la tarjeta obligatoria
   "🔴 ALPACA REAL / Esta operación utilizará dinero real" con símbolo, dirección, cantidad,
   entrada, stop, target, riesgo y R:R, y exige pulsar `[CONFIRMAR OPERACIÓN REAL]`.

---

## 3. Limitaciones reales de la API de Alpaca

El sistema **no inventa capacidades**. Estas restricciones son de Alpaca, no del diseño:

- **No hay clientes asíncronos en `alpaca-py`.** Todos son síncronos; cada llamada se ejecuta en
  `asyncio.to_thread` con timeout, reintentos y backoff exponencial.
- **Acciones fraccionales** solo se permiten en órdenes **a mercado**. Un `qty` fraccional en una
  orden límite se rechaza con `UnsupportedOrderShapeError`.
- **`notional`** solo funciona con órdenes **a mercado** de acciones y nunca junto a `qty`.
- **Las órdenes OCO solo cierran posiciones** (ambas legs reducen exposición).
- **Las legs de take profit solo pueden ser límite** (`TakeProfitRequest` únicamente acepta
  `limit_price`).
- **Un trailing stop NO puede ser leg de un bracket u OCO.** Por eso el Position Protection
  Manager implementa una **transición segura**: coloca y **confirma** un stop de reemplazo
  antes de cancelar las legs existentes. Si ese stop falla, se lanza `ValidationError` y **no se
  cancela nada**.
- **`ClosePositionRequest` exige `qty` o `percentage`.** Si no se indica ninguno, el motor envía
  `percentage="all"`.
- Para obtener las legs de un bracket u OCO hay que consultar con `get_orders(nested=True)`.
- Las órdenes **rechazadas por Alpaca nunca se registran como colocadas**: el sistema solo afirma
  una protección cuando recibe confirmación real con un `order_id`.

---

## 4. Puesta en marcha

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # Linux/macOS

cp .env.example .env      # y rellena TELEGRAM_BOT_TOKEN, ALPACA_* y la allowlist
```

### Variables de entorno

| Variable | Por defecto | Descripción |
|----------|-------------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | Token de @BotFather. Obligatorio. |
| `TELEGRAM_ALLOWED_USER_IDS` | — | IDs numéricos separados por `,` o `;`. **Lista vacía = arranque rechazado.** |
| `TELEGRAM_RATE_LIMIT_PER_MINUTE` | 20 | Peticiones por usuario y minuto. |
| `ACTIVE_TRADING_ENVIRONMENT` | `PAPER` | Entorno activo. **PAPER por defecto.** |
| `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET_KEY` | — | Credenciales PAPER. |
| `ALPACA_PAPER_BASE_URL` | `https://paper-api.alpaca.markets` | |
| `ALPACA_PAPER_DATA_URL` | `https://data.alpaca.markets` | |
| `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY` | — | Credenciales REAL. |
| `ALPACA_LIVE_BASE_URL` | `https://api.alpaca.markets` | |
| `ALPACA_LIVE_DATA_URL` | `https://data.alpaca.markets` | |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/botalpaca.db` | En Fly: `sqlite+aiosqlite:////data/botalpaca.db` |
| `DATABASE_MIGRATE` | `true` | Ejecutar Alembic al arrancar. |
| `DATABASE_AUTO_CREATE` | `true` | Crear tablas faltantes directamente. |
| `TIMEZONE` | `America/New_York` | Zona del mercado. |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Logging estructurado con secretos redactados. |
| `SCAN_INTERVAL_SECONDS` | 60 | Monitor de mercado. |
| `MONITOR_INTERVAL_SECONDS` | 60 | Monitor de posiciones. |
| `RECONCILE_INTERVAL_SECONDS` | 300 | Reconciliación SQLite ↔ Alpaca. |
| `API_TIMEOUT_SECONDS` / `API_MAX_RETRIES` / `API_RETRY_BACKOFF_SECONDS` | 20 / 3 / 1.0 | |
| `RISK_*` | ver `config/settings.py` | 18 límites de riesgo. |
| `STRATEGY_*` | | `MIN_SCORE_TO_EXECUTE=70`, `MIN_BARS_REQUIRED=60`, … |
| `PROTECTION_*` | | `DEFAULT_TRAILING_PERCENT=2.0`, `FALLBACK_STOP_PCT=3.0`, … |
| `MONITORING_*` | | `AUTO_TRADING_ENABLED=false` (por defecto), `OPPORTUNITY_ALERT_MIN_SCORE=75`, … |
| `ANALYTICS_*` | | `MIN_SAMPLE_FOR_SIGNIFICANCE=20`, `MIN_SAMPLE_FOR_SHARPE=30` |
| `SCANNER_*` | | `BENCHMARK=SPY`, universo de ~90 símbolos, ventana de dedup |

Los secretos **solo** se leen de variables de entorno. Nunca se guardan en SQLite ni se
escriben en los logs: el filtro de secretos redacta cualquier clave que parezca
`ALPACA_*_SECRET_KEY`, `*_TOKEN`, `*_PASSWORD`, etc.

---

## 5. Ejecución

### Comprobación de estado (segura, no envía órdenes)

```bash
.\.venv\Scripts\python.exe -m botalpaca --check
```

Devuelve 0 si todo está bien, 2 si falta configuración, 3 si hay otro error. Imprime el
entorno activo, el estado de la base de datos y si las credenciales de cada entorno están
presentes. Es también el `HEALTHCHECK` del contenedor.

### Paper

```bash
# 1. PAPER ya es el valor por defecto; basta con tener ALPACA_PAPER_* en .env
# 2. Arranca el bot
.\.venv\Scripts\python.exe -m botalpaca
```

En Telegram: `/start` → `/modo` (debe mostrar 🟢 ALPACA PAPER) → `/analizar` → `/comprar AAPL`.

### Real

1. Rellena `ALPACA_LIVE_*` en `.env`.
2. Arranca el bot (sigue en PAPER).
3. `/modo` → pulsa "🔴 CAMBIAR A ALPACA REAL" → confirma.
4. El sistema verifica autenticación, consulta la cuenta y muestra equity y buying power.
   **No se envía ninguna orden durante el cambio.**
5. A partir de ahí cada operación muestra la tarjeta de dinero real y exige confirmación
   explícita.

Para volver: `/modo` → "🟢 CAMBIAR A ALPACA PAPER" → confirma.

---

## 6. Comandos de Telegram

| Comando | Qué hace |
|---------|----------|
| `/start` | Bienvenida y estado del bot. |
| `/help` | Índice completo de comandos. |
| `/status` | Salud: base de datos, Alpaca, scheduler, kill switch, entorno. |
| `/modo [PAPER\|REAL]` | Muestra o cambia el entorno (siempre con confirmación). |
| `/analizar [SÍMBOLOS]` | Con símbolos: análisis profundo. Sin ellos: escanea el universo. |
| `/oportunidades` | Últimas oportunidades detectadas y memorizadas. |
| `/posiciones` | Posiciones del entorno activo con su protección. |
| `/ordenes` | Órdenes abiertas del entorno activo. |
| `/cuenta` | Equity, cash, buying power, valor de cartera. |
| `/portfolio` | Exposiciones y P&L no realizado. |
| `/comprar SÍMBOLO [CANTIDAD]` | Propuesta de compra (nunca envía sola: confirma). |
| `/vender SÍMBOLO [CANTIDAD]` | Propuesta de venta. |
| `/cerrar SÍMBOLO` | Cierre de posición con confirmación. |
| `/cancelar` | Descarta la confirmación pendiente sin tocar órdenes. |
| `/modificar ORD_ID ...` | Reemplaza qty / stop / límite. |
| `/riesgo [SÍMBOLO]` | Límites de riesgo y su estado actual. |
| `/stats` | Estadísticas históricas del entorno activo. |
| `/historial` | Operaciones cerradas. |
| `/explicar SÍMBOLO` | **"¿Por qué recomendaste AAPL?"** con estrategia, score, confluencias y resultado. |
| `/aprender` | Recomendaciones no vinculantes del motor estadístico. |
| `/monitor [on\|off]` | Activa o pausa los monitores en segundo plano. |
| `/config` | Configuración efectiva **sin secretos**. |
| `/reconciliar` | Reconciliación manual SQLite ↔ Alpaca. |

Botones inline: Comprar, Vender, Aceptar, Rechazar, Cancelar, Cerrar, Break-even,
Activar trailing, Detalles, Estadísticas, Riesgo.

---

## 7. Gestión de protección

- Stop inicial y take profit (bracket automático).
- **Break-even** con buffer y movimiento **progresivo** a medida que el R avanza.
- **Trailing stop** con percent fijo o derivado del ATR.
- **Transición segura a trailing**: el stop de reemplazo se envía y se **confirma** antes de
  cancelar cualquier leg previa. Si falla, nada se cancela y se informa del error.
- **Time stop** configurable (`PROTECTION_DEFAULT_TIME_STOP_MINUTES`).
- **Nunca se afirma una protección sin confirmación de Alpaca.**
- **Reconciliación permanente**: al arrancar y cada `RECONCILE_INTERVAL_SECONDS` se compara
  SQLite contra las órdenes abiertas de Alpaca; se adoptan posiciones huérfanas, se limpian
  filas obsoletas y se crea un stop de emergencia para cualquier posición desprotegida
  (`PROTECTION_AUTO_PROTECT_MISSING_STOP=true`). Si no hay ATR disponible se usa
  `PROTECTION_FALLBACK_STOP_PCT`.

---

## 8. Estadísticas

Cada señal (aceptada **y** rechazada) y cada operación se guardan con entorno, símbolo,
estrategia, setup, timeframe, indicadores, score, confluencias, entrada, stop, target, salida,
resultado, P&L, P&L %, R múltiple, duración, volumen, ATR, régimen, sector y motivos.

Se agrupa por símbolo, estrategia, setup, timeframe, régimen, sector, banda de score, día y
hora. **El tamaño de muestra se muestra siempre**; por debajo de
`ANALYTICS_MIN_SAMPLE_FOR_SIGNIFICANCE` (20) el resultado se etiqueta como
"muestra pequeña" y no se concluding como significativo. Sharpe solo se calcula con al menos
`ANALYTICS_MIN_SAMPLE_FOR_SHARPE` (30) observaciones.

`/aprender` puede **recomendar** ajustes, pero nunca modifica parámetros críticos por sí solo.

---

## 9. Pruebas

```bash
.\.venv\Scripts\python.exe -m pytest -q            # suite completa
.\.venv\Scripts\python.exe -m pytest --cov=botalpaca -q
.\.venv\Scripts\python.exe -m ruff check .
```

La suite cubre indicadores, análisis, estrategias, scoring y confluencia, riesgo y
dimensionamiento, construcción de órdenes (bracket, OCO, trailing rechazado, stop-limit),
protección (break-even, orden de la transición a trailing, cancelación, reconciliación),
ejecución (barrera de entorno, idempotencia, kill switch, rechazo del broker), seguridad,
Telegram, persistencia, migraciones y **aislamiento PAPER/REAL**.

Ningún test toca la red: el cliente de Alpaca y el de datos están simulados.

---

## 10. Despliegue en Fly.io

```bash
fly volumes create botalpaca_data --size 1 --region iad
fly secrets set TELEGRAM_BOT_TOKEN=... ALPACA_PAPER_API_KEY=... ALPACA_PAPER_SECRET_KEY=...
fly deploy
fly logs
```

- `fly.toml` monta el volumen `botalpaca_data` en `/data` y fija
  `DATABASE_URL=sqlite+aiosqlite:////data/botalpaca.db`, de modo que la base de datos y el
  kill switch sobreviven a los reinicios y despliegues.
- `DATABASE_MIGRATE=true` ejecuta Alembic al arrancar; las tablas se crean antes de que el bot
  atienda comandos.
- La aplicación **no expone ningún puerto HTTP**: la salud se comprueba con
  `python -m botalpaca --check`, que es el `HEALTHCHECK` de la imagen.
- `TZ=America/New_York` para que los horarios coincidan con la sesión.

Alternativa local: `docker compose up -d` (volumen nombrado `botalpaca_data`).

---

## 11. Seguridad

- Allowlist de usuarios de Telegram; **una lista vacía impide el arranque**.
- Confirmación explícita para cada operación; confirmación extra y obligatoria para REAL.
- Kill switch persistido en SQLite (`/config kill switch on|off`), comprobado dentro del motor
  de ejecución.
- Límites de pérdida diaria, semanal y mensual; drawdown máximo; tamaño máximo por posición y
  por exposición total, sectorial y correlacionada.
- Circuit breaker: se abre tras 5 fallos consecutivos del broker.
- Rate limiting por usuario y minuto.
- **Idempotencia**: cada intención de orden se registra con su clave antes de salir; un
  reintento con la misma clave se rechaza con `DuplicateOrderError`.
- Auditoría inmutable de toda orden enviada y recibida (`order_audit`).
- Reconciliación periódica y automática tras reinicios.

---

## 12. Estructura del proyecto

```
botalpaca/
├── domain/         enums, errores, modelos Pydantic
├── config/         settings tipados y logging estructurado
├── db/             modelos SQLAlchemy, sesión async, repositorios, Alembic
├── indicators/     indicadores puros (numpy)
├── analysis/       tendencia, momentum, volumen, volatilidad, estructura, contexto
├── strategies/     12 estrategias + registro
├── confluence/     score 0-100 y explicación de entrada
├── market/         fachada asíncrona de datos de Alpaca
├── execution/      constructor de órdenes, cliente, barrera de entorno
├── portfolio/      cuenta, posiciones, órdenes
├── risk/           dimensionamiento y 18 reglas
├── protection/     stops, break-even, trailing seguro, reconciliación
├── learning/       motor estadístico
├── journal/        trade journal
├── security/       guards y confirmaciones
├── notifications/  renderizadores de tarjetas
├── scanner/        escaneo del universo
├── monitoring/     monitor de mercado y de posiciones
├── telegram/       comandos, teclado y adaptador PTB
├── scheduler/      bucles en segundo plano
├── app.py          contenedor de dependencias
└── __main__.py     punto de entrada
```

---

## 13. Licencia y responsabilidad

Uso educativo. Opera en PAPER primero, con el tamaño mínimo, y entiende cada riesgo antes de
usar dinero real. El autor no se hace responsable de pérdidas.
