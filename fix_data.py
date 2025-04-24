import pandas as pd

# Read the file
print("Reading CSV file...")
df = pd.read_csv('data/SPY_1m.csv')

# Print info about the original file
print(f"Original file has {len(df)} rows")
print(f"First few timestamps: {df['timestamp'].head(3).tolist()}")
print(f"Last few timestamps: {df['timestamp'].tail(3).tolist()}")

# Convert timestamps to UTC and remove timezone info
print("Converting timestamps to UTC...")
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

# Print the converted timestamps
print(f"Converted first few timestamps: {df['timestamp'].head(3).tolist()}")
print(f"Converted last few timestamps: {df['timestamp'].tail(3).tolist()}")

# Save to a new file
output_file = 'data/SPY_1m.csv'
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False)

print("Done!")
