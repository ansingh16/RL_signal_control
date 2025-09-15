from pytris import API
import pandas as pd
import plotly.express as px
import os
import datetime
import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor


api = API()


# Retry wrapper
def fetch_with_retry(site_id, retries=5, delay=5):

    # Date range (last 3 years up to Jan 1, 2025)
    end_date = datetime.datetime(2025, 1, 1)
    delta_time = datetime.timedelta(days=365 * 3)
    start_date = end_date - delta_time
    start_date, end_date = start_date.strftime('%d%m%Y'), end_date.strftime('%d%m%Y')

    daily = api.daily_reports()

    # check if the data already exists
    if os.path.exists(f"../data/raw/df_{site_id}.parquet"):
        print(f"✅ Data already exists for site {site_id}")
    else:
        print(f"⏳ Fetching data for site {site_id}")
        for attempt in range(retries):
            try:
                daily_data = daily.get(
                    sites=site_id,
                    start_date=start_date,
                    end_date=end_date,
                    page_size=5000
                )
                df = pd.DataFrame(daily_data)
                df["site_id"] = site_id
                df.to_parquet(f"../data/raw/df_{site_id}.parquet")
                print(f"✅ Data fetched for site {site_id}")
                return
               
            except Exception as e:
                wait = delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Attempt {attempt+1} failed for site {site_id}: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        
        print(f"❌ Failed completely for site {site_id} after {retries} attempts")
        

if __name__ == "__main__":
    
    new_checkpoints = False
    
    # Initialize API
    sites = api.sites()

    important_highways_around_london = [
        "M25",
        # "M1", "M4", "M40", "M11", "A1(M)", "M3",
        # "A406", "A205", "A2", "A3", "A13", "A12", "A10", "A4"
    ]


    
    if not new_checkpoints:
        print("⏳ Using existing checkpoints")
        check_points_df = pd.read_csv("../data/raw/check_points.csv")
    
    else:
        print("⏳ Generating new checkpoints")
        # Collect checkpoints
        check_points = []
        for road in important_highways_around_london:
            for site in sites.all():
                if road == site.description.split('/')[0]:
                    check_points.append({
                        "name": road,
                        "site_id": site.id,
                        "site_name": site.description.split('/')[1],
                        "latitude": site.latitude,
                        "longitude": site.longitude
                    })

        check_points_df = pd.DataFrame(check_points)
        os.makedirs("../data/raw", exist_ok=True)
        check_points_df = check_points_df.sample(min(100, len(check_points_df)))  # sample up to 100
        check_points_df.to_csv("../data/raw/check_points.csv", index=False)


    print(f"⏳ Fetching data for {len(check_points_df)} sites") 
    
    # for site_id in tqdm.tqdm(check_points_df["site_id"].to_list(), desc="Fetching sites"):
    #     fetch_with_retry(site_id)

    
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(fetch_with_retry, check_points_df["site_id"].to_list())

    print(f"✅ Completed fetching data for {len(check_points_df)} sites")
