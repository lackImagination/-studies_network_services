# Тест backpressure: одновременно отправляем много запросов в /predict-queued.

import asyncio
import sys
from collections import Counter

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400

FARM = {
    "region": "Krasnodar",
    "crop_type": "wheat",
    "area_ha": 120.5,
    "temperature_avg": 18.2,
    "precipitation_mm": 80,
    "payment_delay_days": 45,
    "previous_defaults": 1,
    "debt": 6000000,
}


async def main() -> None:
    limits = httpx.Limits(max_connections=N)

    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        responses = await asyncio.gather(
            *[
                client.post(
                    f"{URL}/predict-queued",
                    json={**FARM, "farm_id": f"BURST-{i}"},
                )
                for i in range(N)
            ],
            return_exceptions=True,
        )

    codes = Counter(
        r.status_code if isinstance(r, httpx.Response) else "error"
        for r in responses
    )

    print(f"Отправлено одновременно: {N}")
    for code, count in sorted(codes.items(), key=lambda x: str(x[0])):
        print(f"  HTTP {code}: {count}")

    rejected = next(
        (
            r for r in responses
            if isinstance(r, httpx.Response) and r.status_code == 429
        ),
        None,
    )

    if rejected is not None:
        print("\nПример отклоненного запроса:")
        print("  тело:", rejected.json())
        print("  Retry-After:", rejected.headers.get("retry-after"))
    else:
        print("\n429 не получено: увеличьте число запросов (второй аргумент).")


asyncio.run(main())
