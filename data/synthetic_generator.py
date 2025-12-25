import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

def generate(start, days, n_stores, n_items, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    rows = []
    # base demand per item
    item_base = np.random.poisson(lam=5, size=n_items) + 1
    item_price = np.round(np.random.uniform(1.0, 20.0, size=n_items), 2)
    # store multipliers
    store_mul = np.random.uniform(0.5, 1.5, size=n_stores)
    for store_idx in range(n_stores):
        for item_idx in range(n_items):
            base = item_base[item_idx] * store_mul[store_idx]
            price = item_price[item_idx]
            stock = int(base * 10)  # initial stock
            for d in dates:
                # sim: promotion occasionally
                on_promo = 1 if np.random.rand() < 0.05 else 0
                # holiday: Sundays simulated
                is_holiday = 1 if d.weekday() == 6 else 0
                # seasonal effect: small weekly pattern
                weekday = d.weekday()
                weekday_mul = 1.0 + 0.15 * (4 - abs(3 - weekday)) / 4.0
                demand_mean = base * weekday_mul * (0.9 if on_promo == 0 else 1.4)
                sales = np.random.poisson(lam=max(0.1, demand_mean))
                # shipments: random deliveries with probability; when below threshold, shipments arrive
                shipments = 0
                if stock < base * 2 and np.random.rand() < 0.25:
                    # receive a shipment with qty around base*10
                    shipments = int(np.random.poisson(lam=base * 10)) + int(base * 5)
                # update stock: arrivals before sales (assume morning)
                stock += shipments
                # actual sales limited by stock (simulate stock-out)
                actual_sales = min(stock, sales)
                stock -= actual_sales
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "store_id": store_idx + 1,
                    "item_id": item_idx + 1,
                    "price": price,
                    "on_promotion": int(on_promo),
                    "is_holiday": int(is_holiday),
                    "shipments_received": shipments,
                    "sales": int(actual_sales),
                    "stock_on_hand": int(stock)
                })
    df = pd.DataFrame(rows)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_data.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--n_stores", type=int, default=10)
    parser.add_argument("--n_items", type=int, default=50)
    args = parser.parse_args()
    df = generate(args.start, args.days, args.n_stores, args.n_items)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows:", df.shape[0])

if __name__ == "__main__":
    main()
