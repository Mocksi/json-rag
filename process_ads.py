import json
import sys
import copy

# Load JSON data
with open('tenant_metrics_imagery_boost.json', 'r') as f:
    data = json.load(f)

# Process the data
new_data = []
for item in data:
    trait_name = item.get('trait', {}).get('name', '')
    has_imagery = 'Imagery' in trait_name
    
    if not has_imagery:
        # Create a deep copy to avoid modifying the original item if it appears multiple times
        processed_item = copy.deepcopy(item)
        for ad in processed_item.get('ads', []):
            metrics = ad.get('metrics', {})
            for key in metrics:
                if isinstance(metrics[key], (int, float)):
                    metrics[key] = metrics[key] / 2
        new_data.append(processed_item)
    else:
        # Append the original item if it has Imagery trait
        new_data.append(item)

# Save the modified data
with open('tenant_metrics_imagery_boost_modified.json', 'w') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("Processing complete. Modified file saved as tenant_metrics_imagery_boost_modified.json") 