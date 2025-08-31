"""
Prepare data for ARCH model comparison
Based on the notebook code
"""

import numpy as np
import pandas as pd
import io
import re

def prepare_quarterly_data():
    """Prepare the quarterly data for ARCH model"""
    
    print("Preparing quarterly data...")
    
    # ----------------------------------------------
    # Real Disposal Income per Capita  & Income Growth
    # ----------------------------------------------
    path = "data/Household Debt Burden/bea_data_quartelry_scaled(million).csv"
    
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()
        
        all_lines = raw_text.splitlines()
        line_idx = [i for i, s in enumerate(all_lines) if s.strip().startswith("Line")]
        
        if len(line_idx) < 2:
            raise ValueError("Could not find the two 'Line' header rows")
        
        start = line_idx[0]
        stop_candidates = [i for i, s in enumerate(all_lines) if s.strip().startswith("Legend/Footnotes")]
        stop = stop_candidates[0] if stop_candidates else len(all_lines)
        
        table_text = "\n".join(all_lines[start:stop])
        df = pd.read_csv(io.StringIO(table_text), sep=",", engine="python", header=None, dtype=str)
        
        years = df.iloc[0, 2:].tolist()
        quarters = df.iloc[1, 2:].tolist()
        periods = [f"{int(y)}Q{str(q).strip()[-1]}" for y, q in zip(years, quarters)]
        vals = df[df[0]=='39'].iloc[0,2:].astype(float).tolist()
        
        result = pd.DataFrame({
            "YYYYQ": periods,
            "RealDPI_perCapita": vals
        }).dropna(subset=["RealDPI_perCapita"]).reset_index(drop=True)
        
        result["IncGrow"] = result["RealDPI_perCapita"].pct_change() * 100.0
        result["Date"] = pd.PeriodIndex(result["YYYYQ"], freq="Q").to_timestamp(how="start")
        result = result[["Date", "YYYYQ", "RealDPI_perCapita", "IncGrow"]]
        
        print(f"✓ Loaded income data: {len(result)} quarters")
        
    except Exception as e:
        print(f"Error loading income data: {e}")
        return None
    
    # -------------------
    # Unemployment Rate
    # -------------------
    try:
        dfm = pd.read_csv("data/Unemployement/unemployment_rates_US_monthly.csv")
        dfm = dfm.rename(columns={"Unnamed: 0": "Year"})
        long = dfm.melt(id_vars="Year", var_name="Month", value_name="Unemp").dropna(subset=["Unemp"])
        long["M"] = long["Month"].str.extract(r"M(\d{2})").astype(int)
        long["Q"] = ((long["M"] - 1) // 3 + 1).astype(int)
        
        q = long.groupby(["Year", "Q"], as_index=False)["Unemp"].mean()
        q["Unemp"] = q["Unemp"].round(2)
        q["YYYYQ"] = q["Year"].astype(int).astype(str) + "Q" + q["Q"].astype(int).astype(str)
        q["Date"] = pd.PeriodIndex(q["YYYYQ"], freq="Q").to_timestamp(how="start")
        q = q[["Date", "YYYYQ", "Unemp"]].sort_values("Date").reset_index(drop=True)
        q["dUnemp"] = q["Unemp"].diff()
        
        print(f"✓ Loaded unemployment data: {len(q)} quarters")
        
    except Exception as e:
        print(f"Error loading unemployment data: {e}")
        q = pd.DataFrame()
    
    # ----------------------------------
    # Financial Obligation Ratio & Housing
    # ----------------------------------
    try:
        df_for_dsr = pd.read_csv("data/Household Debt Burden/Household_Debt_Service_Ratio_quarterly.csv")
        df_for_dsr['observation_date'] = pd.to_datetime(df_for_dsr['observation_date'], errors="coerce")
        
        df2 = pd.read_csv("data/Household Debt Burden/Financial_Obligations_Ratio_quarterly.csv")
        df2['observation_date'] = pd.to_datetime(df2['observation_date'], errors="coerce")
        
        df3 = pd.read_csv("data/Household Debt Burden/Consumer_Debt_Service_Ratio_quarterly.csv")
        df3['observation_date'] = pd.to_datetime(df3['observation_date'], errors="coerce")
        
        df4 = pd.read_csv("data/Housing Price Index/housing_price_index_quarterly.csv")
        df4['observation_date'] = pd.to_datetime(df4['observation_date'], errors="coerce")
        
        df_for_dsr = pd.merge(df_for_dsr, df2, on=["observation_date"], how="outer")
        df_for_dsr = pd.merge(df_for_dsr, df3, on=["observation_date"], how="outer")
        df_for_dsr = pd.merge(df_for_dsr, df4, on=["observation_date"], how="outer")
        
        df_for_dsr = df_for_dsr.rename(columns={"observation_date": "Date"})
        df_for_dsr["YYYYQ"] = df_for_dsr["Date"].dt.to_period("Q").astype(str)
        df_for_dsr = df_for_dsr.sort_values("Date").reset_index(drop=True)
        
        df_for_dsr = df_for_dsr.rename(columns={
            "TDSP": "TotalDSR",
            "CDSP": "ConsumerDSR",
            "FODSP": "FOR",
            "USSTHPI": "HPI",
        })
        
        df_for_dsr["dFOR"] = df_for_dsr["FOR"].diff()
        df_for_dsr["dTotalDSR"] = df_for_dsr["TotalDSR"].diff()
        df_for_dsr["dConsumerDSR"] = df_for_dsr["ConsumerDSR"].diff()
        
        print(f"✓ Loaded debt service ratios: {len(df_for_dsr)} quarters")
        
    except Exception as e:
        print(f"Error loading debt service data: {e}")
        df_for_dsr = pd.DataFrame()
    
    # ----------------------------------
    # NCO
    # ----------------------------------
    try:
        df = pd.read_csv("data/Bank-level Data (Include CC NCO Rate)/credit_card_nco_panel_cleaned.csv", 
                        usecols=["YQ","NCO_RATE_Q", "AVG_CC_LOANS"])
        
        agg = (df.groupby("YQ")
               .apply(lambda g: (g["NCO_RATE_Q"] * g["AVG_CC_LOANS"]).sum() / g["AVG_CC_LOANS"].sum(),
                     include_groups=False)
               .rename("NCO_RATE_Q").to_frame()
               .rename_axis("YYYYQ")
               .reset_index())
        
        agg = agg.sort_values("YYYYQ").reset_index(drop=True)
        agg["NCO_RATE_Q"] = agg["NCO_RATE_Q"] * 100  # Percent
        agg["dNCO"] = agg["NCO_RATE_Q"].diff()
        
        print(f"✓ Loaded NCO data: {len(agg)} quarters")
        
    except Exception as e:
        print(f"Error loading NCO data: {e}")
        agg = pd.DataFrame()
    
    # Merge all data
    base = pd.merge(result, q, on=["Date","YYYYQ"], how="outer")
    base = pd.merge(base, df_for_dsr, on=["Date","YYYYQ"], how="outer")
    base = pd.merge(base, agg, on=["YYYYQ"], how="outer")
    base = base.sort_values("Date").reset_index(drop=True)
    
    # Calculate additional features
    base["AssetVal"] = base["HPI"].astype(float) / base["RealDPI_perCapita"].astype(float) * 100.0
    base["dAssetVal"] = base["AssetVal"].diff()
    base["dHPI"] = base["HPI"].diff()
    
    # Save raw data
    base.to_csv("baseline_model_quarterly_raw.csv", index=False)
    
    # Create clean version with only the features we need
    clean = base[["Date", "YYYYQ", "dNCO", "dUnemp", "IncGrow", 
                  "dFOR", "dTotalDSR", "dConsumerDSR", "dAssetVal", "dHPI"]]
    clean.to_csv("baseline_model_quarterly.csv", index=False)
    
    print(f"✓ Saved baseline_model_quarterly.csv: {len(clean)} quarters")
    
    # Create time-shifted version for ARCH model
    clean_tail = clean.tail(46).copy()
    y = clean_tail.iloc[1:-1, :3].copy()
    y = y.rename(columns={"dNCO": "dNCO_t"})
    
    X = clean_tail.shift(1).iloc[1:-1, 3:].copy()
    X = X.add_suffix("_t-1")
    
    shifted = pd.concat([y, X], axis=1)
    shifted.to_csv("baseline_model_quarterly_time_shifted.csv", index=False)
    
    print(f"✓ Saved baseline_model_quarterly_time_shifted.csv: {len(shifted)} quarters")
    
    return shifted

if __name__ == "__main__":
    df = prepare_quarterly_data()
    if df is not None:
        print("\nData preparation complete!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    else:
        print("\nError: Data preparation failed")