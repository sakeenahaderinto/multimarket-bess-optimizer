import asyncio
import httpx
import pandas as pd
import openmeteo_requests
from datetime import date, timedelta
from pathlib import Path

from config import settings
from ingestion.weather import HOURLY_VARS

START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)


async def backfill_day_ahead() -> None:
    all_records = []
    current = START_DATE
    async with httpx.AsyncClient(timeout=30.0) as client:
        while current <= END_DATE:
            chunk_end = min(current + timedelta(days=9), END_DATE)
            url = (
                f"https://api.electricitymaps.com/v3/price-day-ahead/past-range"
                f"?zone=GB&start={current}T00:00:00.000Z&end={chunk_end}T23:59:00.000Z"
            )
            try:
                response = await client.get(url, headers={"auth-token": settings.em_api_key})
                response.raise_for_status()
                data = response.json()
                records = data.get("data", [data] if "value" in data else [])
                all_records.extend(records)
                print(f"Day-ahead: {current} → {chunk_end} ({len(records)} rows)")
            except Exception as e:
                print(f"Day-ahead: {current} → {chunk_end} failed — {e}")
            current += timedelta(days=10)
            await asyncio.sleep(0.5)
    df = pd.DataFrame(all_records)
    output_path = settings.data_dir / "raw" / "em_day_ahead" / "em_day_ahead_backfill.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Day-ahead backfill complete: {len(df)} rows written")


async def backfill_bmrs() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        current_year = START_DATE.year
        year_records = []

        current = START_DATE
        while current <= END_DATE:
            if current.year != current_year:
                if year_records:
                    df = pd.concat(year_records, ignore_index=True)
                    output_path = settings.data_dir / "raw" / "bmrs" / f"bmrs_{current_year}.parquet"
                    df.to_parquet(output_path, engine="pyarrow")
                    print(f"BMRS: {current_year} written ({len(df)} rows)")
                current_year = current.year
                year_records = []

            date_str = current.strftime("%Y-%m-%d")
            try:
                response = await client.get(
                    f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date_str}?format=json"
                )
                response.raise_for_status()
                data = response.json()
                if data["data"]:
                    year_records.append(pd.DataFrame(data["data"]).convert_dtypes())
            except Exception as e:
                print(f"BMRS: {date_str} failed — {e}")
            current += timedelta(days=1)
            await asyncio.sleep(0.1)

        if year_records:
            df = pd.concat(year_records, ignore_index=True)
            output_path = settings.data_dir / "raw" / "bmrs" / f"bmrs_{current_year}.parquet"
            df.to_parquet(output_path, engine="pyarrow")
            print(f"BMRS: {current_year} written ({len(df)} rows)")


async def backfill_weather() -> None:
    openmeteo = openmeteo_requests.AsyncClient()
    params = {
        "latitude": 51.51,
        "longitude": -0.13,
        "hourly": HOURLY_VARS,
        "timezone": "Europe/London",
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
    }
    responses = await openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    hourly = responses[0].Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )}
    for i, var_name in enumerate(HOURLY_VARS):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(hourly_data)
    output_path = settings.data_dir / "raw" / "weather" / "weather_backfill.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Weather backfill: {len(df)} rows written")


async def backfill_solar() -> None:
    params = {
        "start": f"{START_DATE}T00:00:00",
        "end": f"{END_DATE}T23:30:00",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.pvlive.uk/pvlive/api/v4/gsp/0",
            headers={"Accept-Encoding": "gzip, deflate"},
            params=params,
        )
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["data"], columns=data["meta"])
    output_path = settings.data_dir / "raw" / "solar" / "solar_backfill.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Solar backfill: {len(df)} rows written")


async def backfill_eso() -> None:
    sql = (
        'SELECT * FROM "596f29ac-0387-4ba4-a6d3-95c243140707" '
        'ORDER BY "deliveryStart" ASC '
        'LIMIT 100000'
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            "https://api.neso.energy/api/3/action/datastore_search_sql",
            params={"sql": sql},
        )
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["result"]["records"])
    output_path = settings.data_dir / "raw" / "eso" / "eso_backfill.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    print(f"ESO backfill: {len(df)} rows written")


async def main() -> None:
    print("Starting backfill...")
    await backfill_day_ahead()
    await backfill_weather()
    await backfill_solar()
    await backfill_eso()
    await backfill_bmrs()
    print("Backfill complete.")


if __name__ == "__main__":
    asyncio.run(main())
