import json
import glob
import pandas as pd
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define output directory
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)
data_output_path = output_dir / "processed_truck_data.json"
index_output_path = output_dir / "truck_embeddings.faiss"

def clean_id_component(component):
    """Clean a component of the ID by replacing special characters and spaces with a single underscore."""
    if not component:
        return "Unknown"
    cleaned = re.sub(r'[^\w\s]', ' ', component)
    cleaned = re.sub(r'\s+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

def format_numeric(value):
    """Format numeric values to two decimal places."""
    try:
        return f"{float(value):.2f}" if value != "N/A" else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def map_wheelbase(wheelbase: str) -> str:
    """Map wheelbase measurement to a human-readable label."""
    try:
        wb_value = float(re.search(r'\d+\.?\d*', wheelbase).group())
        if wb_value < 140:
            return "Short Wheelbase"
        elif 140 <= wb_value <= 160:
            return "Long Wheelbase"
        else:
            return "Extra-Long Wheelbase"
    except (ValueError, AttributeError):
        return wheelbase

def map_cab_to_axle(ca: str) -> str:
    """Map cab-to-axle measurement to a human-readable label."""
    try:
        ca_value = float(re.search(r'\d+\.?\d*', ca).group())
        if ca_value == 60:
            return "Standard CA"
        elif ca_value == 84:
            return "Extended CA"
        else:
            return f"CA {ca_value}\""
    except (ValueError, AttributeError):
        return ca

def deduplicate_config_details(config_details: list) -> list:
    """Deduplicate configuration details, keeping the most descriptive version."""
    seen = set()
    deduped = []
    for detail in config_details:
        detail_lower = detail.lower()
        if detail_lower not in seen:
            seen.add(detail_lower)
            deduped.append(detail)
    return deduped

def extract_trim_name(name: str, brand: str, trim: str) -> str:
    """Extract the base trim name based on the brand and predefined trim names."""
    # Define trim names for each make
    trim_names = {
        "GMC": ["AT4", "AT4X", "Denali", "Denali Ultimate", "Pro", "SLE", "SLT"],
        "Chevrolet": ["WT", "Work Truck", "Custom", "LT", "RST", "LTZ", "High Country", "ZR2", "Trail Boss"],
        "Toyota": ["SR", "SR5", "TRD Sport", "TRD Off-Road", "Limited", "Platinum", "Platinum Hybrid", "TRD Pro", "TRD Pro Hybrid"]
    }

    # Get the trim names for the current brand
    brand_trim_names = trim_names.get(brand, [])
    
    # First, check if the trim field (from config["style"]["trimName"]) matches any known trim
    for trim_name in brand_trim_names:
        if trim.lower() == trim_name.lower():
            return trim_name

    # If not found in trim field, search the name field
    for trim_name in brand_trim_names:
        if trim_name.lower() in name.lower():
            return trim_name

    # Default to the trim field if no match is found
    return trim

def flatten_json(json_data, filename):
    try:
        # Access chromeStyle and configuration
        chrome_style = json_data.get("chromeStyle", {})
        config = chrome_style.get("specsSubmodel", {}).get("configuration", {})

        # Extract basic vehicle info
        filename_parts = Path(filename).stem.split("_")
        brand_map = {"toyota": "Toyota", "chevrolet": "Chevrolet", "gmc": "GMC", "ford": "Ford", "ram": "RAM"}
        brand = brand_map.get(filename_parts[1].lower(), filename_parts[1].capitalize()) if len(filename_parts) > 1 else "Unknown"

        # Extract model and trim
        name = chrome_style.get("name", "Unknown").replace("(Natl)", "").strip()
        style_name_without_trim = config.get("style", {}).get("styleNameWithoutTrim", "").replace("(Natl)", "").strip()
        trim = config.get("style", {}).get("trimName", "Unknown").replace("(Natl)", "").strip()

        # Extract the base trim name for the new trim_name field
        trim_name = extract_trim_name(name, brand, trim)

        # Define configuration keywords
        config_keywords = [
            "Crew Cab", "Double Cab", "Regular Cab", "Reg Cab", "CC", "Short Box", "Long Box", "Standard Box",
            "2-Wheel Drive", "4-Wheel Drive", "2WD", "4WD", "4x2", "4x4", "WB", "CA",
            r'\d+\.\d+"', r'\d+"', r'\d+\' \d+"',  # Wheelbase and bed length patterns (e.g., 146", 60", 5'7")
        ]

        # Step 1: Extract configuration details from name
        config_details = []
        wheelbase = None
        cab_to_axle = None
        temp_name = name
        for keyword in config_keywords:
            if isinstance(keyword, str) and keyword not in [r'\d+\.\d+"', r'\d+"', r'\d+\' \d+"']:
                if keyword.lower() in temp_name.lower():
                    config_details.append(keyword)
                    temp_name = temp_name.replace(keyword, "").strip()
            else:
                # Handle regex patterns for wheelbase and bed length
                matches = re.findall(keyword, temp_name)
                for match in matches:
                    # Assume the first match is wheelbase, second is CA (if present)
                    if not wheelbase:
                        wheelbase = match
                        config_details.append(match)
                    elif not cab_to_axle:
                        cab_to_axle = match
                        config_details.append(match)
                    temp_name = temp_name.replace(match, "").strip()

        # Map wheelbase and cab-to-axle to human-readable labels
        if wheelbase:
            wheelbase_label = map_wheelbase(wheelbase)
            config_details = [wheelbase_label if detail == wheelbase else detail for detail in config_details]
        if cab_to_axle:
            ca_label = map_cab_to_axle(cab_to_axle)
            config_details = [ca_label if detail == cab_to_axle else detail for detail in config_details]

        # Step 2: Extract model by removing configuration details
        model = name
        for keyword in config_details:
            model = model.replace(keyword, "").strip()
        # Also remove the original wheelbase and CA measurements
        if wheelbase:
            model = model.replace(wheelbase, "").strip()
        if cab_to_axle:
            model = model.replace(cab_to_axle, "").strip()

        # Remove trim-related keywords (including package levels like "1LT", "2FL", "3SB", "3VL")
        trim_keywords = [trim] + ["LT", "1LT", "2FL", "3SB", "3VL", "WT", "High Country", "Custom", "RST", "LTZ", "Rebel", "Work Truck", "Trail Boss", "Z71", "AT4", "AT4X", "Denali", "Denali Ultimate", "Pro", "SLE", "SLT", "SR", "SR5", "TRD Sport", "TRD Off-Road", "Limited", "Platinum", "Platinum Hybrid", "TRD Pro", "TRD Pro Hybrid"]
        package_level = None
        for keyword in trim_keywords:
            if keyword in model:
                if keyword != trim and keyword in ["1LT", "2FL", "3SB", "3VL"]:  # Identify package level
                    package_level = keyword
                model = model.replace(keyword, "").strip()

        # Clean up extra spaces, commas, and other artifacts in model
        model = re.sub(r'[, ]+', ' ', model).strip()
        if not model:
            model = "Unknown"
        logging.info(f"Extracted model for {filename}: {model}")

        # Step 3: Construct full trim
        full_trim = trim  # Start with the base trim (e.g., "Elevation")
        # Append package level if found (e.g., "3VL")
        if package_level:
            full_trim = f"{full_trim} {package_level}".strip()
        # Ensure configuration details are included in trim (using human-readable labels)
        config_details = deduplicate_config_details(config_details)
        config_part = " ".join(config_details).strip()
        if config_part:
            full_trim = f"{full_trim} {config_part}".strip()
        # Clean up extra spaces in trim
        full_trim = " ".join(full_trim.split())
        logging.info(f"Updated trim for {filename}: {full_trim}")

        year = chrome_style.get("year", "Unknown")
        msrp = chrome_style.get("base_msrp", "N/A")

        # Extract technical specifications
        tech_specs = {spec.get("ConsumerFriendlyTitleName", "Unknown"): spec.get("value", "N/A")
                      for spec in config.get("technicalSpecifications", [])}

        # Extract specific fields with fallbacks
        drive_type = tech_specs.get("Drivetrain", "Unknown")
        tire_size = tech_specs.get("Front Tire Size", "N/A").replace("P", "")
        towing_capacity = tech_specs.get("Maximum Towing Capacity (pounds)", "N/A")
        bed_length = tech_specs.get("Cargo Bed Length (inches)", "N/A")
        engine_displacement = tech_specs.get("Displacement (liters/cubic inches)", "N/A").split('/')[0].replace(' L', 'L')

        # Extract payload capacity
        payload_capacity = tech_specs.get("Maximum Payload Capacity (pounds)", "N/A")
        if payload_capacity == "N/A":
            payload_capacity = next((item.get("description", "") for item in config.get("standardEquipment", [])
                                    if "Maximum Payload" in item.get("description", "")), "N/A")
            payload_capacity = re.search(r'(\d+\.?\d*)#?', payload_capacity).group(1) if payload_capacity != "N/A" else "N/A"
        payload_capacity = format_numeric(payload_capacity)

        # Infer cylinders from engine type
        engine_type = tech_specs.get("Engine Type and Required Fuel", "N/A")
        cylinders = "N/A"
        if "V8" in engine_type:
            cylinders = "8"
        elif "V6" in engine_type or "I6" in engine_type or "V-6" in engine_type:
            cylinders = "6"
        elif "I4" in engine_type:
            cylinders = "4"
        elif "V" in engine_type:
            cylinders = re.search(r'V-?(\d+)', engine_type).group(1) if re.search(r'V-?(\d+)', engine_type) else "N/A"

        # Standardize fuel type
        fuel_type = "GAS"
        if "Diesel" in engine_type:
            fuel_type = "DIESEL"
        elif "Hybrid" in engine_type or "Electric" in engine_type:
            fuel_type = "HYBRID"
        elif "Gas" not in engine_type and "N/A" in engine_type:
            fuel_type = "N/A"

        # Extract and standardize cab configuration
        cab = " ".join([detail for detail in config_details if any(kw in detail.lower() for kw in ["crew cab", "double cab", "regular cab", "reg cab", "cc", "standard ca", "extended ca", "wheelbase"])])
        if not cab:
            cab = style_name_without_trim
        cab = cab.replace("CrewMax", "Crew Cab").replace("Double Cab", "Crew Cab")
        if "5.5' Bed" in cab or "5.5 Bed" in cab:
            cab = "Crew Cab Short Bed"
        elif "6.5' Bed" in cab or "7 foot bed" in cab or "6.5 Bed" in cab:
            cab = "Crew Cab Long Bed"
        elif "6' Bed" in cab or "6 Bed" in cab or "6'4\" Box" in cab:
            cab = "Crew Cab Medium Bed"
        else:
            cab = "Crew Cab"

        # Generate unique doc_id
        doc_id_components = [
            clean_id_component(brand),
            clean_id_component(model),
            clean_id_component(full_trim),
            clean_id_component(str(year)),
            clean_id_component(cab),
            clean_id_component(drive_type)
        ]
        doc_id = "_".join(doc_id_components)

        # Extract all equipment for content
        equipment = [item.get("description", "") for item in config.get("standardEquipment", [])]
        content = "\n".join(equipment)
        if not content:
            content = "No equipment details available."

        # Enrich content with key specs
        content += "\n" + "\n".join([
            f"Towing Capacity: {towing_capacity}",
            f"Engine Displacement: {engine_displacement}",
            f"Tire Size: {tire_size}",
            f"Drive Type: {drive_type}",
            f"Fuel Type: {fuel_type}",
            f"Cab: {cab}",
            f"Payload Capacity: {payload_capacity}"
        ])

        # Standardize fuel economy format
        fuel_economy = tech_specs.get("EPA Fuel Economy, combined/city/highway (mpg)", "N/A")
        is_estimate = False
        if fuel_economy != "N/A":
            logging.info(f"Raw fuel economy for {filename}: {fuel_economy}")
            is_estimate = "(Est)" in str(fuel_economy)
            fuel_economy = re.sub(r'\s*\(?(Est|2024)\)?\s*', '', str(fuel_economy))
            fuel_economy = re.sub(r'\s+', ' ', fuel_economy).strip()
            logging.info(f"Cleaned fuel economy for {filename}: {fuel_economy}")
            content += f"\nFuel Economy Note: {'Estimated' if is_estimate else 'Official EPA'}"

        # Add notes for missing specs
        cca_value = tech_specs.get("Cold Cranking Amps @ 0°F", "N/A")
        logging.info(f"Cold Cranking Amps for {filename}: {cca_value}")
        if str(cca_value).strip().lower() in ["n/a", "none", "", "null"] or cca_value is None:
            content += "\nCold Cranking Amps: Contact Dealer"
        if tech_specs.get("Maximum Alternator Capacity (amps)", "N/A") == "N/A":
            content += "\nAlternator Capacity: Contact Dealer"

        if not content:
            logging.warning(f"Empty content for {filename}")
            return None

        flattened = {
            "id": doc_id,
            "brand": brand,
            "model": model,
            "trim": full_trim,
            "trim_name": trim_name,  # New field for the base trim name
            "year": year,
            "msrp": msrp,
            "drive_type": "4WD" if "Four Wheel Drive" in drive_type else "2WD",
            "tire_size": tire_size,
            "towing_capacity": towing_capacity,
            "bed_length": bed_length,
            "wheelbase": wheelbase if wheelbase else "N/A",
            "cab_to_axle": cab_to_axle if cab_to_axle else "N/A",
            "engine_displacement": engine_displacement,
            "cylinders": cylinders,
            "fuel_type": fuel_type,
            "cab": cab,
            "content": content,
            "Maximum Payload Capacity (pounds)": payload_capacity,
            "As Spec'd Payload Capacity (pounds)": payload_capacity,
            "EPA Fuel Economy, combined/city/highway (mpg)": fuel_economy,
            **{k: v for k, v in tech_specs.items() if k not in ["Curb Weight", "Base Curb Weight (pounds)"]}
        }
        return flattened
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None

def clean_data(df):
    # Additional data cleaning
    df["cylinders"] = df["cylinders"].fillna("N/A")
    df["fuel_type"] = df["fuel_type"].fillna("N/A")
    df["Maximum Payload Capacity (pounds)"] = df["Maximum Payload Capacity (pounds)"].fillna("N/A")
    return df

# Process JSON files
json_files = glob.glob(r"E:\VehicleData\Data\2025\*\*\*.json", recursive=True)
logging.info(f"Found {len(json_files)} inputJSON files")
documents = []
for file in json_files:
    try:
        with open(file, 'r') as f:
            json_data = json.load(f)
            flattened_data = flatten_json(json_data, file)
            if flattened_data:
                documents.append(flattened_data)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file}: {e}")
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

# Save processed data
df = pd.DataFrame(documents)
df = clean_data(df)
df.to_json(data_output_path, orient="records", lines=False)

# Create FAISS index with batch processing
embedder = SentenceTransformer('all-mpnet-base-v2')
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    batch = df['content'][i:i+batch_size].tolist()
    batch_embeddings = embedder.encode(batch, show_progress_bar=True)
    embeddings.append(batch_embeddings)
embeddings = np.vstack(embeddings)
logging.info(f"Embeddings shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
faiss.write_index(index, str(index_output_path))

# Verify index
index = faiss.read_index(str(index_output_path))
logging.info(f"FAISS index dimension: {index.d}, vectors: {index.ntotal}")

logging.info(f"Preprocessing complete. Data saved to '{data_output_path}' and embeddings to '{index_output_path}'.")